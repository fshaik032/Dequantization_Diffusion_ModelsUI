import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import os, sys
from deepfloyd_if.modules.stage_II import IFStageII
import configargparse
from loss import *
from metric_logger import MetricLogger
from datetime import datetime
from tqdm import tqdm
import json
import random
from dataset_palette import *
from pathlib import Path


p = configargparse.ArgParser() 
p.add('--steps', required=False, default='super27', 
      type=str, help='Number of steps for stage inference')
p.add('--tests', required=False, default='r', 
      type=str, help='Specify what image editing task to run (for inference): r for reconstruction, h for hsv aug, p for patch editing, t for palette transfer, m for color map transfer')
p.add('--sample_loop', required=False, default='ddpm',
      type=str, help='ddpm or ddim')
p.add('--paths', required=True, type=str, help='Glob-like pattern containing images to train on')
p.add('--logdir', required=True, help='Where to save results')
p.add('--lr', required=False, default=0.00002, type=float, help='learning rate')
p.add('--aug_level', required=False, default=0.0,
      type=float, help='aug level for stage 2')
p.add('--dynamic_thresholding_p', required=False, default=0.95,
      type=float, help='dynamic thresholding p')
p.add('--dynamic_thresholding_c', required=False, default=1.0,
      type=float, help='dynamic thresholding c')
p.add('--effective_batch_size', required=False,
      default=12, type=int, help='training batch size')
p.add('--model_load_path', required=False, default=None,
      help='pretrained model, None means train from scratch.')
p.add('--data_mode', required=False, default='G',
      choices=('L', 'G', 'T'), help='flags for dataset: "L" for we will condition on Luminance. "G" for gradient. "T" for Thresholded gradient.')
p.add('--eval', action='store_true',
      help='Run in evaluation mode - do not train')
p.add('--post_L', action='store_true',
      help='Force output to take on source luminance')
p.add('--bias_tsteps', action='store_true',default=False,
      help='Bias training tsteps towards super27 timesteps')
p.add('--amp', action='store_true', help='use amp during train and test')
p.add('--partialTrainable', action='store_true', help='only beginning CN layers trainable', default=False)
p.add('--max_val_imgs', required=False, default=30, type=int,
      help='Max images per val run')
p.add('--num_epochs', required=False, default=10,
      type=int, help='number of epochs')
p.add('--seed', required=False, default=42,
      type=int, help='Global seed, for sample variety')
p.add('--aux_channels', required=False, default=7, type=int,
      help='Number of aux channels for ControlNet. Consider padding channels to multiple of 8.')
p.add('--val_res', required=False, default=256, type=int,
      help='Resolution of validation images.')
p.add('--force_aux_8', action='store_true', default=False,
      help='Force number of aux channels input to ControlNet to be multiple of 8 (pad with zeros)')
p.add('--support_noise_less_qsample_steps', required=False, default=0, type=int,
      help='At inference time, add noise to some initial image instead of starting from pure noise (i.e. skip denoising timesteps), 0 means max noise (destroy input render), steps-1 means least amount of added noise')
p.add('--doCN', action='store_true', default=False,
      help='Whether to instantiate a ControlNet module')


configs = p.parse_args()
paths = configs.paths
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(configs.seed) 

def create_dataloader(dataset, batch_size, num_workers, seed=42):
    if dataset.stage == 'train':
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            drop_last=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )

steps = configs.steps
model_load_path = configs.model_load_path
now = datetime.now()  # current date and time

date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
if configs.eval:
    prefix = 'eval_postL-{}_steps-{}_skip-{}_'.format(
        configs.post_L, steps, configs.support_noise_less_qsample_steps)
else:
    prefix = 'train-LR-{}_time_'.format(configs.lr)


log_path_base = os.path.join(
    configs.logdir, f"{configs.data_mode}_"+prefix+date_time)

if not os.path.exists(log_path_base):
    Path(log_path_base).mkdir(parents=True, exist_ok=True)

with open(os.path.join(log_path_base, 'commandline_args.txt'), 'w') as f:
    json.dump(configs.__dict__, f, indent=2)

text_prompts = torch.from_numpy(np.load('empty_prompt_1_77_4096.npz', allow_pickle=True)[
                                'arr']).to('cuda:0').repeat(configs.effective_batch_size, 1, 1)

mpath = 'IF-II-M-v1.0' #small version of DeepFloyd
model_kwargs = {'doCN': configs.doCN, 'aux_ch': configs.aux_channels + (8 - configs.aux_channels % 8) % 8 if configs.force_aux_8 else configs.aux_channels, 'attention_resolutions': '32,16'}

stage_2 = IFStageII(mpath, device='cuda:0', filename=model_load_path, model_kwargs=model_kwargs)


def _setDtype(stage_2):
    stage_2.model.dtype = torch.float32 #tested on float32 mixed precision
    stage_2.model.precision = '32' 
    if configs.doCN:
        stage_2.model.control_model.dtype = stage_2.model.dtype
        stage_2.model.control_model.precision = stage_2.model.precision
    for name, p in stage_2.model.named_parameters():
        p.data = p.type(stage_2.model.dtype)

_setDtype(stage_2)

if configs.force_aux_8:
    zeros_tensor = torch.zeros(
        (configs.effective_batch_size, 8-(configs.aux_channels % 8), configs.training_res, configs.training_res)).to("cuda").type(stage_2.model.dtype)

for name, p in stage_2.model.named_parameters():
    p.requires_grad = False


partialTrainable = configs.partialTrainable #means only train a few conv layers at the head of the CN but not the CN itself, for debugging
params = []

if configs.doCN: #Let the ControlNet module be trainable
    for name, param in stage_2.model.control_model.named_parameters():
        if 'encoder_pooling' in name or 'encoder_proj' in name:
            continue
        if partialTrainable:
            if name != 'input_blocks.0.0.weight' and not name.startswith('zero_convs.') and not name.startswith('input_hint_block.') and not name.startswith('middle_block_out.'):
                continue 
        param.requires_grad = True
        params.append(param)

# Count total trainable parameters in `params`
num_trainable_params = sum(p.numel() for p in params if p.requires_grad)
print(f"Total trainable parameters in ControlNet: {num_trainable_params:,}")

# Count total parameters in stage_2.model.control_model
num_total_params = sum(p.numel() for p in stage_2.model.parameters())
print(f"Total available parameters in whole model: {num_total_params:,}")

# Helper function for image quantization
def quantize_and_convert(out_tensor, target_tensor, num_colors):
    def tensor_to_numpy(tensor):
        # Convert (1,3,H,W) tensor to (H,W,C) numpy array
        return (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    def numpy_to_tensor(array):
        # Convert (H,W,C) numpy array to (1,3,H,W) tensor
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    
    def process_image(image_np):
        pil_image = Image.fromarray(image_np)
        palette_image = pil_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        pal_rgb = palette_image.convert('RGB')
        return np.asarray(pal_rgb)

    # Convert tensors to numpy arrays
    out_np = tensor_to_numpy(out_tensor)
    target_np = tensor_to_numpy(target_tensor)

    # Process images
    out_quantized = process_image(out_np)
    target_quantized = process_image(target_np)

    # Convert back to tensors
    out_tensor_quantized = numpy_to_tensor(out_quantized)
    target_tensor_quantized = numpy_to_tensor(target_quantized)

    return out_tensor_quantized, target_tensor_quantized


def save_labeled_image_grid(images, labels, save_path, grid_size=(2, 5)):
    """
    Save a grid of labeled PyTorch images as a single JPEG.
    
    Args:
    - images: List of PyTorch tensors of shape (1, C, H, W) with range 0-1
    - labels: List of strings corresponding to each image
    - save_path: Path to save the output JPEG
    - grid_size: Tuple (rows, cols) for the grid layout
    """
    assert len(images) == len(labels), "Number of images and labels must match"
    
    # Convert PyTorch tensors to PIL Images
    pil_images = [transforms.ToPILImage()(img.squeeze(0)) for img in images]
    
    # Get dimensions
    img_width, img_height = pil_images[0].size
    rows, cols = grid_size
    
    # Create a new image with space for labels
    margin = 20
    label_height = 30
    grid_width = cols * (img_width + margin) - margin
    grid_height = rows * (img_height + label_height + margin) - margin
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    

    font = ImageFont.truetype("FreeSerif.ttf", 20) 
    draw = ImageDraw.Draw(grid_image)
    
    for i, (img, label) in enumerate(zip(pil_images, labels)):
        row = i // cols
        col = i % cols
        x = col * (img_width + margin)
        y = row * (img_height + label_height + margin)
        
        # Paste the image
        grid_image.paste(img, (x, y))
        
        # Draw the label
        text_width = draw.textlength(label, font=font)
        text_x = x + (img_width - text_width) // 2
        text_y = y + img_height
        draw.text((text_x, text_y), label, fill='black', font=font)
    
    # Save the final image
    grid_image.save(save_path, 'JPEG', quality=100)

# helper function to save out images at test time
def val_helper(epoch, val_loader, image_root, metrics_root, num_colors=16, max_iterations=20, metric_logger=None):
    [log_a, log_b] = metric_logger
    start =  0
    for iteration, sample in enumerate(tqdm(val_loader), start=start):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=configs.amp):
            conditioning = sample['conditioning'].cuda()
            B,C,H,W = conditioning.shape
            color = sample['color'].cuda()
            target = sample['target'].cuda()
            if configs.force_aux_8:
                if C % 8 > 0: #C is the channels dim of conditioning
                    conditioning = torch.cat((conditioning, zeros_tensor), dim=1)

            out, metadata = stage_2.embeddings_to_image(sample_timestep_respacing=str(steps), low_res=2*color-1, support_noise=2*color-1,
                                                            support_noise_less_qsample_steps=configs.support_noise_less_qsample_steps, seed=None, t5_embs=text_prompts[0:1, ...], hint=2*conditioning-1, aug_level=configs.aug_level, sample_loop=configs.sample_loop, dynamic_thresholding_p=configs.dynamic_thresholding_p,dynamic_thresholding_c=configs.dynamic_thresholding_c)

        out = (out + 1)/2
        if configs.post_L: #
            lab = cv2.cvtColor((255*out.squeeze().cpu().numpy().transpose(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab[:,:,0] = (255*sample['L'].numpy()).astype(np.uint8)
            out = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
            out = torch.from_numpy(out.transpose(2,0,1)).float().cuda().unsqueeze(0) / 255.

        metrics = log_a.add(out, target)
        out_quantized, color_quantized = quantize_and_convert(out, color, num_colors=num_colors)
        metrics = log_b.add(out_quantized, color_quantized)
        
        diff = torch.abs(out-target)
        diff = torch.sum(
            diff, dim=1, keepdim=True).repeat(1, 3, 1, 1).clamp(0,1)

        save_path = os.path.join(image_root, str(iteration) + '.jpg')
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)
        directory = os.path.dirname(metrics_root) 
        os.makedirs(directory, exist_ok=True)

        gradx, grady, col_ind, tex_ind = conditioning[:,3:4,...], conditioning[:,4:5,...], conditioning[:,5:6,...], conditioning[:,6:7,...]
        transform = lambda tensors: [t.repeat(1, 3, 1, 1) for t in tensors]
        gradx, grady, col_ind, tex_ind = transform([gradx, grady, col_ind, tex_ind])

        #first row: color, out_quantized, target_quantized, out, target
        #second row: diff, gradx, grady, tex_ind, col_ind

        def _save_tensor_as_image(tensor, filename):            
            # Convert to PIL Image
            img = transforms.ToPILImage()(tensor.squeeze(0))

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save the image
            img.save(filename, quality=100)
        
        if configs.eval:
            images = [color, color_quantized, out_quantized, out, target, diff, gradx, grady, tex_ind, col_ind]
            labels = ['input', 'input_quantized', 'out_quantized', 'generated', 'GT', 'diff', 'grad_x', 'grad_y', 'tex_indicator', 'col_indicator']   
            for s_idx in range(len(images)):
                _save_tensor_as_image(images[s_idx],os.path.join(image_root, str(iteration), labels[s_idx]+'.jpg'))
                

        gradx, grady, tex_ind, col_ind, out_quantized = map(lambda y: F.pad(y, (5, 5, 5, 5)), [
                                            gradx, grady, tex_ind, col_ind, out_quantized])
        diff, color, out, target, color_quantized = map(lambda y: F.pad(y, (5, 5, 5, 5)), [
                                            diff, color, out, target, color_quantized])

        images = [color, color_quantized, out_quantized, out, target, diff, gradx, grady, tex_ind, col_ind]
        labels = ['input', 'input_quantized', 'out_quantized', 'generated', 'GT', 'diff', 'grad_x', 'grad_y', 'tex_indicator', 'col_indicator']

        if val_loader.dataset.do_HSV_aug:
            if configs.eval:
                for myStr in ['quantized_source_image', 'quantized_augmented_image', 'source_image']:
                    _save_tensor_as_image(sample[myStr].permute(0,3,1,2),os.path.join(image_root, str(iteration), myStr+'.jpg'))

        if val_loader.dataset.do_color_transfer:
            if configs.eval:
                for myStr in ['dst_rgb', 'dst_pal']:
                    _save_tensor_as_image(sample[myStr].permute(0,3,1,2),os.path.join(image_root, str(iteration), myStr+'.jpg'))

            dst_rgb, dst_pal = map(lambda y: F.pad(y, (5, 5, 5, 5)), [
                                            sample['dst_rgb'].permute(0,3,1,2), sample['dst_pal'].permute(0,3,1,2)])
            images.insert(5,dst_rgb)
            labels.insert(5,'style image')
            images.append(dst_pal)
            labels.append('style palette')
            save_labeled_image_grid(images, labels, save_path, grid_size=(2, 6))
        else:
            save_labeled_image_grid(images, labels, save_path, grid_size=(2, 5))

        if iteration == start + max_iterations:
            break

    val_epoch_info = log_a.getEpochInfo()
    print('epoch info: {}'.format(epoch))
    print(val_epoch_info)
    print('\n[RGB error] epoch: {}, time: {}'.format(epoch, datetime.now().strftime(
        "%m-%d-%Y-%H-%M-%S")), file=open(metrics_root, 'a'))
    print('WRITING METRICS')
    print(val_epoch_info, file=open(metrics_root, 'a'))
    print('\n[Palette error]', file=open(metrics_root, 'a'))
    val_epoch_info = log_b.getEpochInfo()
    print(val_epoch_info, file=open(metrics_root, 'a'))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# run validation: make dataloaders, and for each one, get metrics and images
def runVal(tot_iteration, stage='val'):
    stage_2.model.eval()
    
    # Metric loggers
    log_a = MetricLogger(gamma=1.0)  # RGB error
    log_b = MetricLogger(gamma=1.0)  # Palette error
    metric_loggers = [log_a, log_b]
    
    # A small helper function to streamline repeated steps
    def evaluate_dataset(
        tot_iter,
        stage,
        dataset_kwargs,
        subfolders,        # list or tuple of folder names for storing outputs
        txt_filename,      # text filename
        loggers,
        max_iters,
        worker_count=2,
    ):
        """
        Creates a dataset, dataloader, and calls val_helper. 
        Resets the provided metric loggers afterward.
        """
        val_dataset = MyDataset(**dataset_kwargs)
        val_loader  = create_dataloader(
            val_dataset, batch_size=1, num_workers=worker_count, seed=configs.seed
        )
        
        # Build image directory and text path
        img_dir = os.path.join(
            log_path_base, f"{stage}_imgs", str(tot_iter), *subfolders
        )
        txt_path = os.path.join(
            log_path_base, f"{stage}_imgs", *subfolders[:-1], txt_filename
        )
        
        # Call val_helper
        val_helper(
            tot_iter,
            val_loader,
            img_dir,
            txt_path,
            metric_logger=loggers,
            max_iterations=max_iters,
            # Pass num_colors if provided in the dataset kwargs
            num_colors=dataset_kwargs.get("colors", None)
        )
        
        # Reset metrics
        for lg in loggers:
            lg.reset_metrics()

    # For each texture option 
    for tex in [False, True]:
        
        # --- HSV Augmentation ---
        if 'h' in configs.tests:
            # Evaluate multiple HSV strengths and color counts
            for aug in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for numColors in [4, 16, 64]:
                    ds_kwargs = {
                        "paths": paths,
                        "mode": configs.data_mode,
                        "stage": stage,
                        "colors": numColors,
                        "do_palette": True,
                        "do_texture": tex,
                        "do_color_transfer": False,
                        "patch_random_color": False,
                        "size": configs.val_res,
                        "do_HSV_aug": True,
                        "augmentation_strength": aug,
                    }
                    evaluate_dataset(
                        tot_iteration,
                        stage,
                        ds_kwargs,
                        subfolders=[f"hsv_aug_{aug}", f"{numColors}_tex_{tex}"],
                        txt_filename=f"{numColors}_tex_{tex}.txt",
                        loggers=metric_loggers,
                        max_iters=configs.max_val_imgs,
                    )

        # --- Color Transfer ---
        if 't' in configs.tests:
            for method in ['color', 'neg']:
                for numColors in [8, 32]:
                    ds_kwargs = {
                        "paths": paths,
                        "mode": configs.data_mode,
                        "stage": stage,
                        "colors": numColors,
                        "do_palette": True,
                        "do_texture": tex,
                        "do_color_transfer": True,
                        "patch_random_color": False,
                        "transfer_method": method,
                        "size": configs.val_res,
                    }
                    evaluate_dataset(
                        tot_iteration,
                        stage,
                        ds_kwargs,
                        subfolders=["transfer", method, f"{numColors}_tex_{tex}"],
                        txt_filename=f"{numColors}_tex_{tex}.txt",
                        loggers=metric_loggers,
                        max_iters=configs.max_val_imgs,
                    )

        # --- Reconstruction ---
        if 'r' in configs.tests:
            for numColors in [64, 16, 4]:
                ds_kwargs = {
                    "paths": paths,
                    "mode": configs.data_mode,
                    "stage": stage,
                    "colors": numColors,
                    "do_palette": True,
                    "do_texture": tex,
                    "do_color_transfer": False,
                    "patch_random_color": False,
                    "size": configs.val_res,
                }
                evaluate_dataset(
                    tot_iteration,
                    stage,
                    ds_kwargs,
                    subfolders=[f"recon_tex_{tex}_{numColors}", str(numColors)],
                    txt_filename=f"recon_tex_{tex}_{numColors}.txt",
                    loggers=metric_loggers,
                    max_iters=configs.max_val_imgs,
                )

        # --- Patch-based Editing ---
        if 'p' in configs.tests:
            for rand in [False, True]:
                ds_kwargs = {
                    "paths": paths,
                    "mode": configs.data_mode,
                    "stage": stage,
                    "colors": 8,
                    "do_palette": False,
                    "do_texture": tex,
                    "do_color_transfer": False,
                    "patch_random_color": rand,
                    "size": configs.val_res,
                }
                evaluate_dataset(
                    tot_iteration,
                    stage,
                    ds_kwargs,
                    subfolders=[f"patch_tex_{tex}_rand_{rand}"],
                    txt_filename=f"patch_tex_{tex}_rand_{rand}.txt",
                    loggers=metric_loggers,
                    max_iters=configs.max_val_imgs,
                )

        # --- Colormap Transfer ---
        if 'm' in configs.tests:
            for numColors in [64, 16, 4]:
                for cmap in ['viridis', 'plasma', 'inferno', 'cividis', 
                             'bone', 'magma', 'hsv', 'twilight']:
                    ds_kwargs = {
                        "paths": paths,
                        "mode": configs.data_mode,
                        "stage": stage,
                        "colors": numColors,
                        "do_palette": True,
                        "do_texture": True,
                        "do_color_transfer": True,
                        "patch_random_color": False,
                        "size": configs.val_res,
                        "transfer_method": 'cmap',
                        "cmap": cmap,
                    }
                    evaluate_dataset(
                        tot_iteration,
                        stage,
                        ds_kwargs,
                        subfolders=["cmap", cmap, f"{numColors}_with_grad"],
                        txt_filename=f"{numColors}.txt",
                        loggers=metric_loggers,
                        max_iters=configs.max_val_imgs,
                    )


if configs.eval:
    runVal(0,stage='test')
    import sys
    print('eval only, no training.')
    sys.exit()

def worker_init_fn(worker_id):
    # Set the seed for the random module
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

train_dataset = MyDataset(paths=paths, mode=configs.data_mode, stage='train')
train_loader = create_dataloader(train_dataset, batch_size=configs.effective_batch_size, num_workers=8, seed=50)

# TODO: consider pure BF16 train/inference, using torch.compile, 8-bit Adam, switch from memory-efficient attention to Flash Attn. 
opt = torch.optim.AdamW(params, lr=float(configs.lr))

epoch = 0
gd = stage_2.get_diffusion(None)

print_freq = 50
val_every = 8000 #iterations between running validation, 8000
scaler = torch.cuda.amp.GradScaler(enabled=configs.amp)
total_iterations = 0

def save_checkpoint_and_val():
    #Run validation - configs.tests sets which modes to run during validation
    runVal(total_iterations, stage='val')  
    ckptPath = os.path.join(log_path_base, 'ckpt.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': stage_2.model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, ckptPath)

for epoch in range(configs.num_epochs):
    stage_2.model.train()
    print('starting epoch: ', epoch)
    tot_loss = 0

    for iteration, sample in enumerate(tqdm(train_loader)):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=configs.amp):
            conditioning = sample['conditioning'].cuda()
            B,C,H,W = conditioning.shape
            color = sample['color'].cuda()
            target = sample['target'].cuda()
            if configs.force_aux_8:
                if C % 8 > 0: #C is the channels dim of conditioning
                    conditioning = torch.cat((conditioning, zeros_tensor), dim=1)
            # low-res is the RGB palette image or region to inpaint. 
                    
            # x_start will be the GT image
            loss = gd.training_losses(stage_2.model, x_start=2*target-1,  model_kwargs={
                                    'text_emb': text_prompts, 'low_res': 2*color-1, 'hint': 2*conditioning-1, 'aug_level': configs.aug_level}, bias_tsteps=configs.bias_tsteps)

            l = loss['loss'].mean()
        scaler.scale(l).backward() 
        total_iterations += 1
        tot_loss += l.item()
        if total_iterations % print_freq == print_freq-1:
            print('epoch {}, iteration {}, cur training loss: {}'.format(
                epoch, total_iterations, tot_loss/print_freq))
            tot_loss = 0
            sys.stdout.flush()

        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

        if total_iterations % val_every == 0:
            # TODO consider saving multiple checkpoints
            save_checkpoint_and_val()
save_checkpoint_and_val()