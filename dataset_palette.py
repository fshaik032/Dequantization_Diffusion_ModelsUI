import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from skimage.measure import label
from PIL import Image
import random
from skimage.color import rgb2gray
from scipy import ndimage
import matplotlib
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
import glob


class Config:
    MAX_COLORS = 128
    IMAGE_SIZE = 256
    VAL_SIZE = 100
    TEST_SIZE = 1000
    SEED = 42

class MyDataset(Dataset):
    def __init__(self, paths: str, mode: str = 'G', stage: str = 'train', colors=4, do_palette=True, do_texture=True, do_color_transfer=False, patch_random_color=True, size=256, seed=50, transfer_method='color', cmap='magma', blend=1, do_HSV_aug=False, augmentation_strength=0):
        """
        Initialize the dataset.
        
        :param mode: 'L' for luminance, 'G' for gradient, 'T' for thresholded gradient
        :param stage: 'train', 'val', or 'test'
        """
        self.paths = glob.glob(paths)
        if len(self.paths) == 0:
            print(f"Error: No files found matching pattern: {paths}")
            import sys
            sys.exit(1)
        self.config = Config()
        self.mode = mode
        self.stage = stage
        assert stage in ["train", "test", "val"]
        self.data = self._load_and_split_data()
        self.seed = seed
        # Create a separate RNG for each major operation
        # can likely be simplified
        self.main_rng = np.random.default_rng(self.seed)
        self.crop_rng = np.random.default_rng(self.seed + 1)
        self.texture_rng = np.random.default_rng(self.seed + 2)
        self.color_rng = np.random.default_rng(self.seed + 3)

        self.colors=colors
        self.do_palette=do_palette
        self.do_texture=do_texture
        self.do_color_transfer=do_color_transfer #To implement
        self.patch_random_color=patch_random_color
        self.config.IMAGE_SIZE = size
        self.col_values = list(range(8)) #2^0 ... 2^7 colors in the palette
        min_weight = 1.5
        max_weight = 0.5
        # probabilities for selecting number of colors in the palette, bias towards fewer colors because it's harder
        self.col_weights = [min_weight - (min_weight - max_weight) * (value / 7) for value in self.col_values]
        self.col_weights = self.col_weights/np.sum(self.col_weights)
        self.seed = seed if stage != "train" else None #seed is none during training, so it's different upon restart
        self.np_rng = np.random.default_rng(self.seed)
        assert transfer_method in ['color', 'freq', 'int', 'neg', 'cmap']
        self.transfer_method=transfer_method
        self.cmap = cmap
        self.blend = blend
        self.do_HSV_aug = do_HSV_aug
        self.augmentation_strength = augmentation_strength

    def _load_and_split_data(self) -> List[str]:
        """
        Load image paths from the gzip file and split the data based on the stage.
        
        :return: List of image paths for the current stage
        """
        paths = self.paths
        
        rng = random.Random(self.config.SEED)
        rng.shuffle(paths)      
        
        if self.stage == 'train':
            return paths[self.config.VAL_SIZE + self.config.TEST_SIZE:]
        elif self.stage == 'val':
            return paths[:self.config.VAL_SIZE]
        elif self.stage == 'test':
            return paths[self.config.VAL_SIZE:self.config.VAL_SIZE + self.config.TEST_SIZE]
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single item from the dataset.
        
        :param idx: Index of the item
        :return: Dictionary containing processed image data
        """
        image_path = self.data[idx]
        target = self._load_and_preprocess_image(image_path) #RGB, 0-255 after this function
        
        # Store the luminance before normalizing the target
        luminance = self._compute_luminance(target.copy()) #L channel is 0-1 float
        
        control, dst_pal, dst_rgb, quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image = self._create_control_channels(target.copy(), luminance)

        if self.stage != "train" and self.do_HSV_aug:
            target = augmented_image_rgb
            mlist = [quantized_source_image, quantized_augmented_image, source_image]
            mlist = list(map(self._normalize_target, mlist))   
            mlist = self._permute_channels(*mlist)

        target = self._normalize_target(target) #0-255 to 0-1
        target, control = self._permute_channels(target, control)


        return {
            "target": target,
            "conditioning": control,
            "color": np.copy(control[:3, :, :]),
            "L": luminance,
            "dst_pal": dst_pal,
            "dst_rgb": dst_rgb,
            "source_image": source_image,
            "quantized_augmented_image": quantized_augmented_image,
            "quantized_source_image": quantized_source_image
        }

    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess the image."""
        target = cv2.imread(str(image_path))
        target = self._resize_and_crop(target)
        if self.stage == 'train':
            if self.np_rng.random() > 0.5:
                target = np.fliplr(target)
            target = self._apply_color_transformations(target)
        return cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    def _resize_and_crop(self, image: np.ndarray) -> np.ndarray:
        """Resize and crop the image to the desired size."""
        height, width = image.shape[:2]
        if height < self.config.IMAGE_SIZE and width < self.config.IMAGE_SIZE:
            # Calculate the largest multiple of 16 that is less than or equal to the height and width
            new_height = (height // 16) * 16
            new_width = (width // 16) * 16
            # Crop the image to the new height and width
            return image[:new_height, :new_width, :] #During inference, make it work on any aspect ratio

        size = self.crop_rng.integers(self.config.IMAGE_SIZE, min(height, width) + 1)

        scale = float(size) / min(height, width)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        max_x = image.shape[1] - self.config.IMAGE_SIZE
        max_y = image.shape[0] - self.config.IMAGE_SIZE
        x = self.np_rng.integers(0, max_x + 1)
        y = self.np_rng.integers(0, max_y + 1)
        
        return image[y:y + self.config.IMAGE_SIZE, x:x + self.config.IMAGE_SIZE]

    def _apply_color_transformations(self, image: np.ndarray) -> np.ndarray:
        """Apply random color transformations to the image."""
        toss = self.np_rng.random()
        if toss < 0.1:
            return self._convert_to_grayscale(image)
        elif toss < 0.2:
            return self._augment_saturation(image)
        return image

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert the image to grayscale."""
        methods = [
            lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            lambda: cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0],
            lambda: np.mean(image, axis=2).astype(np.uint8)
        ]
        gray = self.np_rng.choice(methods)()
        return np.stack([gray, gray, gray], axis=-1)

    def _augment_saturation(self, image: np.ndarray) -> np.ndarray:
        """Augment the saturation of the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_weight = self.np_rng.random() * 2
        hsv[:, :, 1] = np.clip(s_weight * hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _create_control_channels(self, image: np.ndarray, luminance: np.ndarray) -> np.ndarray:
        """Create control channels for the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gradients = None if self.mode == 'L' else np.gradient(gray)
        height, width = image.shape[:2]
        control = np.ones((height, width, 7), dtype=np.float32) #7 channels in the conditioning
        grad_channels = self._get_gradient_channels(gradients, luminance)
        grad_channels, augmented_image, indicators, dst_pal, dst_rgb, quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image = self._get_texture_channels(image, grad_channels)
        control[:, :, :3] = augmented_image / 255.0
        control[:, :, 3:5] = grad_channels
        control[:, :, 5:] = indicators
        return control, dst_pal, dst_rgb, quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image

    def _get_gradient_channels(self, gradients: Tuple[np.ndarray, np.ndarray], luminance: np.ndarray) -> np.ndarray:
        """Get gradient channels based on the mode."""
        if self.mode == 'G':
            return np.stack([(g + 255) / (2 * 255.) for g in gradients], axis=-1)
        elif self.mode == 'L':
            return np.stack([luminance, luminance], axis=-1)
        elif self.mode == 'T':
            return np.stack([np.greater(np.abs(g), 8).astype(np.float32) for g in gradients], axis=-1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def create_segment_mask_optimized(self, labeled_array):
        # Get the total number of segments
        num_segments = np.max(labeled_array) + 1
        
        # Determine the number of segments to select (1 to 15% of total)
        max_segments = max(1, int(0.15 * num_segments))
        num_segments_to_select = self.np_rng.integers(1, max_segments + 1)
        
        # Initialize the mask and the set of selected segments
        mask = np.zeros_like(labeled_array, dtype=bool)
        selected_segments = set()
        
        # Pick a random starting segment
        start_segment = self.np_rng.integers(0, num_segments)
        selected_segments.add(start_segment)
        mask |= (labeled_array == start_segment)
        
        # Pre-compute neighbor relationships
        neighbor_dict = {}
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
        for segment in range(num_segments):
            segment_mask = (labeled_array == segment)
            dilated = ndimage.binary_dilation(segment_mask, kernel)
            neighbor_mask = dilated & ~segment_mask
            neighbors = set(np.unique(labeled_array[neighbor_mask])) - {segment}
            neighbor_dict[segment] = neighbors
        
        # Expand selection until we reach the target number of segments
        all_neighbors = set()
        while len(selected_segments) < num_segments_to_select:
            # Update all neighbors
            new_neighbors = set()
            for segment in selected_segments:
                new_neighbors.update(neighbor_dict[segment])
            all_neighbors = (all_neighbors | new_neighbors) - selected_segments
            
            # If no more neighbors, break the loop
            if not all_neighbors:
                break
            
            # Randomly select a neighbor
            new_segment = self.np_rng.choice(list(all_neighbors))
            selected_segments.add(new_segment)
            all_neighbors.remove(new_segment)
            mask |= (labeled_array == new_segment)
        
        return mask

    def sort_colors(self, color_array):
        # Ensure the color values are in the range [0, 255] and the correct data type
        color_array = np.clip(color_array, 0, 255).astype(np.uint8)
        
        # Reshape the array to a 2D array with a single row
        reshaped_array = color_array.reshape((-1, 1, 3))
        
        # Convert RGB to HSV
        hsv_colors = cv2.cvtColor(reshaped_array, cv2.COLOR_RGB2HSV)
        
        # Reshape back to a 2D array
        hsv_colors = hsv_colors.reshape(-1, 3)
        
        # Sort primarily by hue, then by saturation, then by value
        sorted_indices = np.lexsort((hsv_colors[:, 2], hsv_colors[:, 1], hsv_colors[:, 0]))
        
        # Apply the sorting to the original RGB array
        sorted_colors = color_array[sorted_indices]
        
        return sorted_colors

    def compare_hsv_augmentation(self, pil_image):
        # Load and prepare the image
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Generate random parameters for HSV augmentation
        h_shift = np.random.uniform(-self.augmentation_strength, self.augmentation_strength) * 180
        s_scale = np.random.uniform(1 - self.augmentation_strength, 1 + self.augmentation_strength)
        v_scale = np.random.uniform(1 - self.augmentation_strength, 1 + self.augmentation_strength)

        # Function to apply HSV augmentation with stored parameters
        def augment_hsv(img):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            h = (h + h_shift) % 180
            s = np.clip(s * s_scale, 0, 255)
            v = np.clip(v * v_scale, 0, 255)
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 1. Source image
        source_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # 2. Quantized source image
        pil_quantized_source = pil_image.convert('P', palette=Image.ADAPTIVE, colors=self.colors)
        quantized_source_image = np.array(pil_quantized_source.convert('RGB'))

        # Get the initial palette
        initial_palette = np.array(pil_quantized_source.getpalette()[:3*self.colors]).reshape(self.colors, 3)

        # 3. Augmented source image
        augmented_image = augment_hsv(cv_image)
        augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)

        # 4. Quantized version of the augmented image
        pil_augmented = Image.fromarray(augmented_image_rgb)
        pil_quantized_augmented = pil_augmented.convert('P', palette=Image.ADAPTIVE, colors=self.colors)
        quantized_augmented_image = np.array(pil_quantized_augmented.convert('RGB'))

        # 5. Result of applying augmentation to the color palette of the source image
        palette_bgr = cv2.cvtColor(initial_palette.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_RGB2BGR)
        augmented_palette_bgr = augment_hsv(palette_bgr)
        augmented_palette = cv2.cvtColor(augmented_palette_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        # Apply the augmented palette to the quantized original image
        pil_augmented_palette = Image.new('P', (1, 1))
        pil_augmented_palette.putpalette(augmented_palette.flatten())
        pil_augmented_image = pil_quantized_source._new(pil_quantized_source.im)
        pil_augmented_image.putpalette(augmented_palette.flatten())
        augmented_palette_image = np.array(pil_augmented_image.convert('RGB'))

        return augmented_palette_image, quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image

    def make_palette_transfer(self, palette_image, srcInt): #image and Pal are HWC 0-255 RGB images 
        srcPalette3 = np.array(palette_image.getpalette()[0:3*self.colors]).reshape(self.colors,3)/255.
        # Get the dimensions of pal_rgb
        height, width = srcInt.shape[:2]
        fullColor = np.zeros((height,width,3),dtype=np.float32)
        if self.transfer_method == 'cmap': #for color map, we're doing most similar color
            mapQuery = np.arange(0,1 + (1/(2.*self.colors)),1./(self.colors-1))
            cmap = matplotlib.colormaps[self.cmap]
            cm = cmap(mapQuery)[:,0:3]
            D = np.exp(-distance_matrix(srcPalette3, cm, p=2))

            # Create the discrete palette
            discrete_indices = np.repeat(np.arange(self.colors), height // self.colors)
            if len(discrete_indices) < height:
                discrete_indices = np.pad(discrete_indices, (0, height - len(discrete_indices)), mode='edge')
            discrete_palette = cm[discrete_indices] 
            dst_pal = np.repeat(discrete_palette[:, np.newaxis, :], width, axis=1)

            # Create the smoothed color map
            smoothed_indices = np.linspace(0, 1, height)
            smoothed_colormap = cmap(smoothed_indices)[:, :3]
            dst_rgb = np.repeat(smoothed_colormap[:, np.newaxis, :], width, axis=1)

        else:

            idx = self.np_rng.integers(len(self.data))
            image_path = self.data[idx] 
            target = self._load_and_preprocess_image(image_path) #RGB, 0-255 after this function
            pil_target = Image.fromarray(target)
            targInt = pil_target.convert('P', palette=Image.ADAPTIVE, colors=self.colors)
            targPalette3 = np.array(targInt.getpalette()[0:3*self.colors]).reshape(self.colors,3)/255.
            targCol = np.array(targInt.convert('RGB')) / 255.
            targInt = np.array(targInt)

            if self.transfer_method == 'color':
                D = np.exp(-distance_matrix(srcPalette3, targPalette3,p=2))

            elif self.transfer_method == 'neg':
                D = np.exp(distance_matrix(srcPalette3, targPalette3,p=2))

            elif self.transfer_method == 'freq':
                sortedSrc, countsSrc = np.unique(srcInt, return_counts=True)
                sortedTarg, countsTarg = np.unique(targInt, return_counts=True)
                A = np.reshape(countsSrc/float(np.sum(countsSrc)),(self.colors,1))
                B = np.reshape(countsTarg/float(np.sum(countsTarg)),(self.colors,1))
                D = np.exp(-distance_matrix(A, B,p=2))

            elif self.transfer_method == 'int':
                D = np.exp(-distance_matrix(np.sum(srcPalette3,1,keepdims=True), np.sum(targPalette3,1,keepdims=True),p=2))
                
            # Create an array of row indices
            row_indices = np.arange(height)
            # Calculate which color each row should be (integer division)
            color_indices = row_indices * self.colors // height
    
            # Use advanced indexing to create the image array
            targPalette3 = self.sort_colors(targPalette3*255) / 255.
            image_array = targPalette3[color_indices][:, np.newaxis, :]
            
            # Repeat the colors across all columns
            dst_pal = np.repeat(image_array, width, axis=1)

            dst_rgb = target
                        
        _, matching = linear_sum_assignment(-D)

        for colIDX in range(self.colors):
            mask = srcInt == colIDX
            if self.transfer_method == 'cmap':
                getColor = cm[matching[colIDX],:]
            else:
                getMask = targInt==matching[colIDX]
                getColor = targCol[getMask][0] #get one instance of this color
            fullColor[mask] = self.blend * getColor + (1-self.blend) * srcPalette3[colIDX]

        return (fullColor*255).astype(np.uint8), dst_pal, dst_rgb #palettized src image, palettized style, style image unmodified

    def _get_texture_channels(self, image: np.ndarray, grad_channels: np.ndarray) -> np.ndarray:
        """Get texture channels for the image."""
        height, width = image.shape[:2]
        pil_image = Image.fromarray(image)
        num_colors = 2**self.np_rng.choice(self.col_values, p=self.col_weights) if self.stage == 'train' else self.colors
        palette_image = pil_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        pal_rgb = palette_image.convert('RGB')
        pal_rgb = np.asarray(pal_rgb)
        indexed_palette = np.array(palette_image)
        dst_pal, dst_rgb = 0, 0
        quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image = 0, 0, 0, 0
        if self.stage!="train" and self.do_color_transfer:
            pal_rgb, dst_pal, dst_rgb = self.make_palette_transfer(palette_image, indexed_palette) 

       
        texture_channel = np.ones((height, width))
        color_indicator = np.full((height, width), num_colors / (2 * self.config.MAX_COLORS))
        do_palette_coloring = (self.stage == 'train' and self.np_rng.random() > 0.5) or (self.stage != 'train' and self.do_palette)
        remove_texture = (self.stage == 'train' and self.np_rng.random() > 0.35) or (self.stage != 'train' and not self.do_texture)
        if do_palette_coloring:  #palette-based coloring 
            if remove_texture:
                texture_channel = np.zeros((height, width))
            augmented_image = pal_rgb

            if self.stage!="train" and self.do_HSV_aug:
                augmented_palette_image, quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image = self.compare_hsv_augmentation(pil_image)
                augmented_image = augmented_palette_image

        else: #patch-based recoloring            
            if self.texture_rng.random() > 0.1:
                ft_image = image / 255.
                choice = self.np_rng.integers(0,4)
                if choice == 0:
                    components = felzenszwalb(ft_image, scale=100, sigma=0.5, min_size=50) #int64 from 0-N-1 (n segments total)
                elif choice == 1:
                    components = slic(ft_image, n_segments=250, compactness=10, sigma=1, start_label=0)
                elif choice == 2:
                    components = quickshift(ft_image, kernel_size=3, max_dist=6, ratio=0.5)
                elif choice == 3:
                    gradient = sobel(rgb2gray(ft_image))
                    components = watershed(gradient, markers=250, compactness=0.001)
                mask = self.create_segment_mask_optimized(components)
            else:
                components = label(indexed_palette, connectivity=2, background=-1)
                num_components = np.max(components) + 1 #assume components is 0 : N-1
                num_selecting = self.np_rng.integers(0, num_components)

                selecting = self.np_rng.choice(num_components, size=num_selecting, replace=False)
                mask = np.isin(components, selecting)


            augmented_image, texture_channel, color_indicator = self._apply_patch_based_conditioning(mask, image, texture_channel, color_indicator, remove_texture)

        grad_channels[texture_channel < 0.5] = 0
        return grad_channels, augmented_image, np.stack([color_indicator, texture_channel], axis=-1), dst_pal, dst_rgb, quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image

    def _apply_patch_based_conditioning(self, mask: np.ndarray, image: np.ndarray, 
                                        texture_channel: np.ndarray, color_indicator: np.ndarray, remove_texture: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Apply patch-based conditioning to the image."""
        if np.any(mask):
            for c in range(3):
                image[mask, c] = self.color_rng.integers(0,255) if self.stage != 'train' and self.patch_random_color else np.mean(image[mask, c])
            if remove_texture:
                texture_channel[mask] = 0
        color_indicator[mask] = 1 / (2 * self.config.MAX_COLORS)
        color_indicator[~mask] = 1.

        return image, texture_channel, color_indicator

    def _compute_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the luminance channel of the image.
        
        :param image: Input image with values in range [0, 255]
        :return: Luminance channel with values in range [0, 1]
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0] / 255.0

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        """
        Normalize the target image to [0, 1].
        
        :param target: Input image with values in range [0, 255]
        :return: Normalized image with values in range [0, 1]
        """
        return (target.astype(np.float32) / 255.0)

    def _permute_channels(self, *arrays: np.ndarray) -> List[np.ndarray]:
        """Permute the channels of the arrays."""
        return [np.transpose(arr, (2, 0, 1)) for arr in arrays]