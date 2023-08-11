# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from dataset_tool import convert_dataset


# ----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank,
                                                 world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
                                                 world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)  # The whole function training_loop is to train the model.


# ----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)
    # Creating an output folder
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} epochs')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    os.makedirs(os.path.join(c.run_dir, r'result_image'), exist_ok=True)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    # torch.multiprocessing.set_start_method('spawn')
    ctx = torch.multiprocessing.get_context("spawn")
    print('CPU threads num:', torch.multiprocessing.cpu_count())
    with tempfile.TemporaryDirectory() as temp_dir:  # 这里是多线程处理
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


# ----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True,
                                         max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


# ----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------
# Here are the definitions of the parameters
@click.command()
# Required.
@click.option('--cfg', help='Base configuration', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']),
              default='stylegan3-t', required=True)
@click.option('--gpus', help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), default=1,
              required=True)
@click.option('--aug', help='Augmentation mode', type=click.Choice(['noaug', 'ada', 'fixed']), default='ada',
              show_default=True)
@click.option('--freezed', help='Freeze first layers of D', metavar='INT', type=click.IntRange(min=0), default=0,
              show_default=True)
# Misc hyperparameters.
@click.option('--p', help='Probability for --aug=fixed', metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.2, show_default=True)
@click.option('--target', help='Target value for --aug=ada', metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.6, show_default=True)
@click.option('--batch-gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1))
@click.option('--cbase', help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=16384,
              show_default=True)  # The default is 32768, which works a little better, but the training time is slow!
@click.option('--cmax', help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512,
              show_default=True)
@click.option('--glr', help='G learning rate  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0))
@click.option('--dlr', help='D learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.002,
              show_default=True)
@click.option('--map-depth', help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group', help='Minibatch std group size', metavar='INT', type=click.IntRange(min=1), default=2,
              show_default=True)
# Misc settings.
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list,
              default='none', show_default=True)  # The default is 'fid50k_full' which is very time consuming!
# @click.option('--epochs', help='Total training duration', metavar='epochs', type=click.IntRange(min=1), default=25000,
#               show_default=True)
@click.option('--tick', help='How often to print progress', metavar='epochs', type=click.IntRange(min=1), default=1,
              show_default=True)
@click.option('--snap', help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32', help='Disable mixed-precision', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--nobench', help='Disable cuDNN benchmarking', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=3,
              show_default=True)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
def main(**kwargs):
    print(r'pytorch version:', torch.__version__)
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip  --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --epochs=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

'''---------------------------The following is the definition of the parameter section---------------------------------'''
#the tuning parameter only needs to be changed here, but not anywhere else

    # (1) Parameters of data preprocessing
    source = r'C:/work/data/1.original_dataset/type1/bad'  # Paths to datasets that have not been preprocessed
    data = r'C:/work/data/1.original_dataset/type1/bad.zip'  # Path to the preprocessed dataset (preferably with a zip suffix)
    outdir = r'C:/work/models/3.StyleGAN/1.bad-50+20'  # Output result save path (including intermediate process images and intermediate weights)
    max_images = None  # Maximum number of images to output (preferably set to None!)
    resolution = (128, 128)  # The resolution of the output image (can only be several times 2, such as 64, 128, 256, 512, 1024, etc., the most commonly used is 128 resolution, suitable for RTX2070 or 3060 almost the same performance of the GPU)
    transform = None  # Data augmentation( None center-crop center-crop-wide) pick one
    
    # (2) Parameters of training
    epochs = 20  # Number of iterations
    batch = 2  # The training batch, if it is too big it may report a memory CUDA error, just turn it down!
    gpus = 1  # How many GPUs are used for training
    gamma = 12.2  # Default 8.2. 128 resolution is recommended to be set to 12.2
    snap = 1  # How many iterations per iteration, save the result and intermediate process once
    cond = False  # Whether to use conditional counter generation, default is False
    mirror = True  # Whether it needs to be flipped horizontally
    
    # Whether to load the model to continue training. Set to None if not loaded
    resume = None 
    # resume = r'C:/work/models/3.StyleGAN/1.bad-50/00000-stylegan3-t-good-gpus1-batch2-gamma12.2/network-snapshot-000050.pkl'

'''---------------------------Above is the definition parameter section------------------------------------------------'''
    # Initialize config.
    if os.path.exists(data) is False:
        convert_dataset(source, data, max_images, transform, resolution)  # pylint: disable=no-value-for-parameter
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(),
                                 mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=data)
    if cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = cond
    c.training_set_kwargs.xflip = mirror
    # assign values to parameters, etc.
    # Hyperparameters & settings.
    c.num_gpus = gpus
    c.batch_size = batch
    c.batch_gpu = opts.batch_gpu or batch // gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (
        8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = epochs
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException(
            '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    # Constructing the loss function
    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9  # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2  # Enable path length regularization.
        c.G_reg_interval = 4  # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True  # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1  # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2  # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True  # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10  # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32  # Fade out the blur during the first N epochs.
    # Constructing Image Enhancement
    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1,
                                           scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                                           hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p
    # Whether or not breakpoints are required
    # Resume.
    if resume is not None:
        c.resume_pkl = resume
        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0  # Disable blur rampup.
    # Whether semi-precision training is required
    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    # Start training
    # Launch.
    launch_training(c=c, desc=desc, outdir=outdir, dry_run=opts.dry_run)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
