# CIL Course Project 2021: Road Segmentation

## Abstract

A major part of recent research in the field of semantic image segmentation directs attention to the development of
slight architecture variations of increasing complexity. In this work, we analyze the impact of architecture
modifications of a U-Net, namely the GC-DCNN and other self-developed variations. We find that the exact model
architecture results in solely minor changes in prediction accuracy. Comparing this to other factors in the modeling
pipeline, we conclude that the greatest impact can be reached by using a large and diverse training set. Considering the
minor contribution of model architecture alternations, we propose two post-processing techniques. Albeit not improving
prediction accuracy significantly, these procedures are able to improve some of the predicted roads visually.

## Findings & Results 

### Data

The ETH training data comprises only 100 train images and 94 test images. The train image are not only different in size (400x400
vs. 608x608) from the ETH test data but are also of different quality. Comparing the mean histograms we find that the
test images are missing a lot of color intensity values. These facts increase the complexity of the problem and make it harder to
generalize from the train images to the test images.
<p>
<img src="./other/analysis/histograms/ETH test images_mean_hist.png" alt="mean-test-histograms" height="300" >
<img src="./other/analysis/histograms/ETH train images_mean_hist.png" alt="mean-train-histograms" height="300">
</p>

Since we only have 100 training images we increased our training set by:
- creating augmented ETH images. For that we flipped the original image and stored it separately. Further we saved the
  rotated versions (by 90, 180, 270 degrees) of the original and the flipped image. This increases the training set from
  100 images to a total of 800 images. (see [ETH-dataset](./data/training/eth_dataset))
- using additional training data from Google Maps: [GMaps-public](./data/training/gmaps_public), [GMaps-custom](./data/training/gmaps_custom)

### Models
We employed two models the [U-Net](http://arxiv.org/abs/1505.04597) and
the [GC-DCNN](https://www.sciencedirect.com/science/article/pii/S0020025520304862). To evaluate the influence of the
architecture we additionally adapt both models to improve the predictive results. In the following we refer to the **
U-Net plus** as the U-Net where we increased the pool kernel size from 2 to 4 resulting in a significant improvement.
The **GC-DCNN plus** refers to a deeper version of the original GC-DCNN and can be viewed as a novel combination of the
GC-DCNN with the modules [Atrous Spatial Pyramid Pooling](https://arxiv.org/abs/1606.00915v2) (used as a bridge replacing
the Pyramid Pooling Module) and the [Attention gate](https://arxiv.org/abs/1804.03999v3) (used in the upwards branch).

### Results
- the largest factor was contributed by using more data
- the model architecutre as well as the postprocessing played an important but in comparision a minor factor

## Project Code Structure

```
+-- cil-road-segmentation
   +-- data                      [contains the training data]
   +-- ...
   +-- road_segmentation_main
       +-- configurations        [folder that contains the training parameters in form of *.jsonc files]
       +-- source                [contains the main code to train and run the models]
       +-- train.py              [script to run a training]
       +-- inference.py          [script to predict on the test data]
       +-- ensemble.py           [script that creats out of mutliple preditions an ensemble prediction by averaging]

```

## Reproducibility

- python version: 3.8.5
- cuda: 10.1.243
- cudnn: 7.6.4
- gcc 6.3.0
- python library version according to [requirements](requirements.txt) file

### Setup on leonhard cluster

1. Clone `git clone git@github.com:FrederikeLuebeck/cil-road-segmentation.git`
2. Setup environment
    - Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
    - Create virtual env and install packages:
        - `cd ./cil-road-segmentation/`
        - `python -m venv my_venv`
        - `source ./my_venv/bin/activate`
        - `pip install -r ./road_segmentation_main/requirements.txt`
3. Create a `.env` file in the folder `cil-road-segmentation/road_segmentation_main`
    - `cd road_segmentation_main/`
    - `vim .env`
        ```
        DATA_COLLECTION_DIR=../data/training
        OUTPUT_DIR=trainings
        ```
    - On the leonhard cluster it is advisable to use the scratch as output directory, due to space constraints of the
      home directory. For instance use
      `OUTPUT_DIR=/cluster/scratch/<username>/cil_trainings`.

##

### General: loading environment

1. `cd ./cil-road-segmentation/`
2. Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
3. Load python env: `source ./my_venv/bin/activate`
4. If you want to work with tmux, start tmux with `tmux` (see tmux guide below)

### Run training

1. Load environment
2. Navigate to the road segmentation folder `cd road_segmentation_main/`
3. Edit the configuration file to your needs
    - `vim ./configurations/default.jsonc`
4. Run job on GPU
    - shorter
      run: `bsub -n 2 -J "training-job" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python train.py --configuration ./configurations/default.jsonc'`
        - change the configuration file name if you use a different one `--configuration ./configurations/default.jsonc`
    - longer run with larger
      dataset: `bsub -n 4 -J "long-run" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/default.jsonc'`
    - check job status `bbjobs -w`
    - peek stdout log `bpeek` or `bpeek -f` to continuously read the log
5. Find your training results with `ls ./trainings/`

#### Reproducibility

We used following base submission command on the Leonhard cluster which selects enough cpu memory as well as the 2080Ti
GPU.

```
bsub -n 4 -J "description" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python <....>'
```

To get reproducible results we fixed the random seeds of `torch`, `random` and `numpy` at various points in the code.
Additionally, we set `torch.backends.cudnn.deterministic = True` as suggested on the official pytroch reproducibility
page: https://pytorch.org/docs/1.9.0/notes/randomness.html.

The results of the U-Net are 100% reproducible. The GC-DCNN lacks exact reproducibility because the pyramid pooling
module (PPM) uses the pytorch function `F.interpolate`
which is not numerically stable (as in pytroch version 3.8.5). As a result we evaluated how much the validation accuracy
varies, by running 10 runs of the GC-DCNN baseline using the "experiments dataset".

- min validation accuracy: 0.9724 (removed one outlier with validation accuracy of 0.9715)
- max validation accuracy: 0.9732

##### Baselines

For the baselines we use the "experiments dataset" which is a combination of the ETH and the GMaps-public dataset with a
predefined train and validation split. The images of the GMaps-public dataset were center cropped to size 400x400 to
match the ETH dataset image size.

| Model | Command |
| ----- | ------- |
| U-Net |`bsub -n 4 -J "gcdcnn_exp_baseline" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/gcdcnn_exp_baseline.jsonc'`|
| GC-DCNN |`bsub -n 4 -J "unet_exp_baseline" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/unet_exp_baseline.jsonc'`|

##### Final

For our final submission we used the datasets: ETH, GMaps-public, GMaps-custom with a validation split of 20%.

| Description | Command |
| ----------- | ------- |
| U-Net<br>Augmentations: SSR, RC|`bsub -n 4 -J "unet_final" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/unet_final.jsonc'`|
| U-Net+<br>Augmentations: SSR, RC|`bsub -n 4 -J "unet_final_plus" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/unet_final_plus.jsonc'`|
|GC-DCNN<br>Augmentations: SSR, RC, GN|`bsub -n 4 -J "gcdcnn_final" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/gcdcnn_final.jsonc'`|
|GC-DCNN+<br>Augmentations: SSR, RC, GN|`bsub -n 4 -J "gcdcnn_final_plus" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/gcdcnn_final_plus.jsonc'`|

### Run submission

1. Load environment
2. Navigate to the road segmentation folder `cd road_segmentation_main/`
3. Run job on GPU
    - `bsub -n 1 -J "submission-job" -W 0:05 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/20210415-181009-default'`
        - change the run folder name to the trainings' folder created during
          training: `--run_folder ./trainings/20210415-181009-default`
4. Find the submission file in your run folder
