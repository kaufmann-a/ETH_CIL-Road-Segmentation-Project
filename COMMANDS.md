Here we list commands to reproduce intermediate results that are part of the report.

#### Dataset experiments
| Dataset | Command |
| ------- | ------- |
| ETH |`python train.py --configuration configurations/experiments/datasets/unet_exp_eth.jsonc`|
| GMaps-public |`python train.py --configuration configurations/experiments/datasets/unet_exp_eth_gmaps_public.jsonc`|
| GMaps-custom |`python train.py --configuration configurations/experiments/datasets/unet_exp_gmaps_custom.jsonc`|
| ETH + GMaps-public + custom|`python train.py --configuration configurations/experiments/datasets/unet_exp_eth_gmaps_public_gmaps_custom.jsonc`|

#### U-Net augmentations experiments
| Augmentations | Command |
| ------------- | ------- |
| - |`bsub -n 4 -J "0000_unet_exp_augmentation" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/augmentations/unet/0000_unet_exp_augmentation.jsonc'`|
| ShiftScaleRotate (SSR) |`bsub -n 4 -J "0004_unet_exp_augmentation" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/augmentations/unet/0004_unet_exp_augmentation.jsonc'`|
| SSR+ChanelShuffle |`bsub -n 4 -J "0403_unet_exp_augmentation" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/augmentations/unet/0403_unet_exp_augmentation.jsonc'`|
| SSR+RandomContrast (RC) |`bsub -n 4 -J "0409_unet_exp_augmentation" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/augmentations/unet/0409_unet_exp_augmentation.jsonc'`|
| SSR+RC+GaussNoise |`bsub -n 4 -J "040908_unet_exp_augmentation" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/augmentations/unet/040908_unet_exp_augmentation.jsonc'`|

#### U-Net architecture experiments
| Architecture | Command |
| ------------- | ------- |
| Dilation (3) |`bsub -n 4 -J "unet_exp_dilation" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation.jsonc'`|
| Dilation (18) |`bsub -n 4 -J "unet_exp_dilation_large" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation_large.jsonc'`|
| Filtersize (5) |`bsub -n 4 -J "unet_exp_filter" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/architecture/unet/unet_exp_filter.jsonc'`|
| PoolKernel (4) |`bsub -n 4 -J "unet_exp_stride_pool" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/architecture/unet/unet_exp_stride_pool.jsonc'`|
| Dilation (3), lr = 0.0001 |`bsub -n 4 -J "unet_exp_dilation_lowlr" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation_lowlr.jsonc'`|
