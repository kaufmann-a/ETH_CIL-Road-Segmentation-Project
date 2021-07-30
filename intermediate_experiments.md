Here we list commands to reproduce intermediate results that are part of the report.

### Dataset experiments

Refers to "Table I: Results of training on different data sets (U-Net)" of the report

| Dataset | Command |
| ------- | ------- |
| ETH |`python train.py --configuration configurations/experiments/datasets/unet_exp_eth.jsonc`|
| GMaps-public |`python train.py --configuration configurations/experiments/datasets/unet_exp_eth_gmaps_public.jsonc`|
| GMaps-custom |`python train.py --configuration configurations/experiments/datasets/unet_exp_gmaps_custom.jsonc`|
| ETH + GMaps-public + custom|`python train.py --configuration configurations/experiments/datasets/unet_exp_eth_gmaps_public_gmaps_custom.jsonc`|

### U-Net and GC-DCNN augmentations experiments

Refers to "Table II: Results of Augmentation Experiments for U-Net and GC-DCNN" of the report

|Model | Augmentations | Command |
| ----- | ------------ | ------- |
| U-Net | - |`python train.py --configuration configurations/experiments/augmentations/unet/0000_unet_exp_augmentation.jsonc`|
| U-Net | ShiftScaleRotate (SSR) |`python train.py --configuration configurations/experiments/augmentations/unet/0004_unet_exp_augmentation.jsonc`|
| U-Net | SSR+RandomContrast (RC) |`python train.py --configuration configurations/experiments/augmentations/unet/0409_unet_exp_augmentation.jsonc`|
| U-Net | SSR+RC+GaussNoise |`python train.py --configuration configurations/experiments/augmentations/unet/040908_unet_exp_augmentation.jsonc`|
| GC-DCNN | - |`python train.py --configuration configurations/experiments/augmentations/gcdcnn/0000_gcdcnn_exp_augmentation.jsonc`|
| GC-DCNN | ShiftScaleRotate (SSR) |`python train.py --configuration configurations/experiments/augmentations/gcdcnn/0004_gcdcnn_exp_augmentation.jsonc`|
| GC-DCNN | SSR+RandomContrast (RC) |`python train.py --configuration configurations/experiments/augmentations/gcdcnn/0409_gcdcnn_exp_augmentation.jsonc`|
| GC-DCNN | SSR+RC+GaussNoise |`python train.py --configuration configurations/experiments/augmentations/gcdcnn/040908_gcdcnn_exp_augmentation.jsonc`|

### U-Net architecture experiments

Refers to "Table IV: Results of U-Net Architecture Alterations"

| Architecture | Command |
| ------------- | ------- |
| Dilation (3) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation.jsonc`|
| Dilation (18) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation_large.jsonc`|
| Filtersize (5) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_filter.jsonc`|
| PoolKernel (4) |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_stride_pool.jsonc`|
| Dilation (3), lr = 0.0001 |`python train.py --configuration configurations/experiments/architecture/unet/unet_exp_dilation_lowlr.jsonc`|

### GC-DCNN architecture experiments

Refers to "Table V: Results of GC-DCNN Architecture Alterations"

| Architecture | Command |
| ------------ | ------- |
| Attention | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_attention.jsonc` |
| ASPP | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_aspp_avg_pool.jsonc` |
| ASPP + Attention | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_aspp_avg_pool_attention.jsonc` |
| Deep | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_deep.jsonc` |
| ASPP + Deep | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_deep_aspp_avg_pool.jsonc` |
| Attention + Deep | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_deep_attention.jsonc` |
| ASPP + Attention + Deep | `python train.py --configuration configurations/experiments/architecture/gcdcnn/gcdcnn_exp_deep_aspp_avg_pool_attention.jsonc` |
