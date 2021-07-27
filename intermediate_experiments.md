Here we list commands to reproduce intermediate results that are part of the report.

### Dataset experiments
| Dataset | Command |
| ------- | ------- |
| ETH |`python train.py --configuration configurations/experiments/datasets/unet_exp_eth.jsonc`|
| GMaps-public |`python train.py --configuration configurations/experiments/datasets/unet_exp_eth_gmaps_public.jsonc`|
| GMaps-custom |`python train.py --configuration configurations/experiments/datasets/unet_exp_gmaps_custom.jsonc`|
| ETH + GMaps-public + custom|`python train.py --configuration configurations/experiments/datasets/unet_exp_eth_gmaps_public_gmaps_custom.jsonc`|

### U-Net augmentations experiments
| Augmentations | Command |
| ------------- | ------- |
| - |`python train.py --configuration configurations/experiments/augmentations/unet/0000_unet_exp_augmentation.jsonc`|
| ShiftScaleRotate (SSR) |`python train.py --configuration configurations/experiments/augmentations/unet/0004_unet_exp_augmentation.jsonc`|
| SSR+ChanelShuffle |`python train.py --configuration configurations/experiments/augmentations/unet/0403_unet_exp_augmentation.jsonc`|
| SSR+RandomContrast (RC) |`python train.py --configuration configurations/experiments/augmentations/unet/0409_unet_exp_augmentation.jsonc`|
| SSR+RC+GaussNoise |`python train.py --configuration configurations/experiments/augmentations/unet/040908_unet_exp_augmentation.jsonc`|

### U-Net architecture experiments
| Architecture | Command |
| ------------- | ------- |
| Dilation (3) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation.jsonc`|
| Dilation (18) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation_large.jsonc`|
| Filtersize (5) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_filter.jsonc`|
| PoolKernel (4) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_stride_pool.jsonc`|
| Dilation (3), lr = 0.0001 |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation_lowlr.jsonc`|
