# CIL Course Project 2021: Road Segmentation

## Abstract

A major part of recent research in the field of semantic image segmentation directs attention to the development of
slight architecture variations of increasing complexity. In this work, we analyze the impact of architecture
modifications of a U-Net, namely the GC-DCNN and other self-developed variations. We find that the exact model
architecture results in solely minor changes in prediction accuracy. Comparing this to other factors in the modeling
pipeline, we conclude that the greatest impact can be reached by using a large and diverse training set. Considering the
minor contribution of model architecture alternations, we propose two post-processing techniques. Albeit not improving
prediction accuracy significantly, these procedures are able to improve some of the predicted roads visually.

_**TODO: Add image**_

## Findings & Results

### Data

The ETH training data comprises only 100 train images and 94 test images. The train image are not only different in
size (400x400 vs. 608x608) from the ETH test data but are also of different quality. Comparing the mean histograms we
find that the test images are missing a lot of color intensity values. These facts increase the complexity of the
problem and make it harder to generalize from the train images to the test images.
<p>
<img src="./other/analysis/histograms/ETH test images_mean_hist.png" alt="mean-test-histograms" height="100%" >
<img src="./other/analysis/histograms/ETH train images_mean_hist.png" alt="mean-train-histograms" height="100%">
</p>

Since we only have 100 training images we increased our training set by:

- creating augmented ETH images. For that we flipped the original image and stored it separately. Further we saved the
  rotated versions (by 90, 180, 270 degrees) of the original and the flipped image. This increases the training set from
  100 images to a total of 800 images. (see [ETH-dataset](./data/training/eth_dataset))
- using additional training data from Google Maps: [GMaps-public](./data/training/gmaps_public)
  , [GMaps-custom](./data/training/gmaps_custom)

### Models

We employed two models the [U-Net](http://arxiv.org/abs/1505.04597) and
the [GC-DCNN](https://www.sciencedirect.com/science/article/pii/S0020025520304862). To evaluate the influence of the
architecture we additionally adapt both models to improve the predictive results. In the following we refer to the
**U-Net plus** as the U-Net where we increased the pool kernel size from 2 to 4 resulting in a slight improvement.
The **GC-DCNN plus** refers to a deeper version of the original GC-DCNN and can be viewed as a novel combination of the
GC-DCNN with the modules [Atrous Spatial Pyramid Pooling](https://arxiv.org/abs/1606.00915v2) (used as a bridge
replacing the Pyramid Pooling Module) and the [attention gate](https://arxiv.org/abs/1804.03999v3) (used in the upwards
branch).

### Postprocessing

We implemented different postprocessing techniques to make the quality of predictions better.
1. Classical methods:
    - Repeated dilation followed by same number of erosions
    - Median filtering
   The filter size and the type was too dependent on the kind of image, so instead of hand tuning these we looked for machine learning based solutions.  
2. Retrain on binary:
    We used the best predictions of the U-Net & GCDCNN as a training set and used it to retrain the network to learn to connect roads by joining lines and remove noisy predictions.
    - U-Net with partial convolution: We replace the normal convolutions with partial convolution layers in UNET. This gave sharper and denoised predictions compared to normal UNET.
    - Increasing the receptive field: We tried experiments with increasing dilation which improved connectivity of disjoint segments.
3. Learning hough transforms:
   We nudge the network towards predicting connected roads by explicitly presenting possible connected line fragments.
 

### Results

- the largest factor was contributed by using more data
- the model architecture as well as the postprocessing played an important but in comparison a minor factor

_**TODO: add short text maybe copy conclusion of report?**_

## Project Code Structure

Below we give a short non-exhaustive overview of the different folders and files together with their usage.

```
+-- cil-road-segmentation
   +-- data                      [contains the training data]
   +-- ...
   +-- road_segmentation_main
       +-- configurations        [contains the training parameters in form of *.jsonc files]
       +-- source                [contains the main code to train and run the models]
       +-- train.py              [script to run a training]
       +-- inference.py          [script to predict on the test data]
       +-- ensemble.py           [script that creats out of mutliple preditions an ensemble prediction by averaging]
       +-- ...
```

Our code is build such that it allows to
1. Reproduce runs
2. Compare runs
3. Keep results of finished runs

We use configuration files to not only run different models with different configurations but also to
reproduce past runs. Configuration files can be found in the
folder `cil-road-segmentation/road_segmentation_main/configurations`. They allow to change the dataset, data
augmentations, model, model parameters, optimizer, learning rate scheduler, and so on. Moreover, logging with
`tensorboard` and `comet` gives us the ability to track and compare results of different runs at ease. For every run a
"run-folder" is created which takes the name `<datetime>-<configfile-name>`. This folder keeps the `stdout` log,
the `tensorboard` log and additionally the model weights-checkpoint (see [Training folder structure](#training-folder-structure). 
This folder serves as a back up of executed runs.

`train.py`:
This is the main script to run a training. The main commandline argument is `--configuration` which contains the
configuration file path.

`inference.py`: This script helps to get model predictions using the ETH test dataset. The main commandline argument
is `--run_folder` which takes the path to the "run-folder" created during training. Then this script will automatically
load the best model checkpoint and create the submission.csv file inside the "run-folder" in the
folder `prediction-<datetime>`.

### Training folder structure
```
+-- trainings
    +-- <datetime>-<configfile_name>      [this is a training "run-folder"]
        +-- prediction-<datetime>         [contains the model predictions and the submission file]
        |   +-- <configfile>              [copy of the configuration file used in inference]
        |   +-- submission.csv            [the submission file to hand in predictions]
        +-- tensorboard                   [contains the tensorboard log]
        |   +-- events.out.tfevents.*
        +-- weights_checkpoint            [contains the model checkpoints]
        |   +-- <epoch>_*.pth             [model checkpoint file]
        |   +-- ...
        +-- <configfile>                  [copy of the configuration file used in training]
        +-- logs.txt                      [contains the stdout log]
```

`weights_checkpoint`: There are two model weights checkpoints. The interval based checkpoint files
called `<epoch>_checkpoint.pth` created after a curtain number of epochs and the files `<epoch>_best.pth` created
whenever the model achieves a new best validation accuracy.

## Reproducibility

Here we list the used library versions, which are loaded and or installed when following the steps below (and working
with the leonhard cluster).

- python version: 3.8.5
- cuda: 10.1.243
- cudnn: 7.6.4
- gcc 6.3.0
- python library version according to [requirements](./road_segmentation_main/requirements.txt) file

### 1. Initial setup on leonhard and environment installation

1. Clone this git repository `git clone git@github.com:FrederikeLuebeck/cil-road-segmentation.git`
2. Environment setup
    - Load the leonhard software modules:  `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
    - Create a virtual environment and install the required python packages:
        - `cd ./cil-road-segmentation/`
        - `python -m venv cil_venv`
        - `source ./cil_venv/bin/activate`
        - `pip install -r ./road_segmentation_main/requirements.txt`

### 2. Add environment variables

3. Create a file called `.env` in the folder `cil-road-segmentation/road_segmentation_main`. This file should contain
   the configuration of the data collection directory as well as the output directory.
    - `cd road_segmentation_main/`
    - `vim .env`
4. Add the following environment variables to the file:
    - `DATA_COLLECTION_DIR`=Path to the training data
    - `OUTPUT_PATH`=Path to which the training runs (model checkpoints etc.) should be saved
    - For instance:
        ```
        DATA_COLLECTION_DIR=../data/training
        OUTPUT_DIR=trainings
        ```
    - On the leonhard cluster it is advisable to use the scratch as output directory, due to space constraints of the
      home directory. For instance use
      `OUTPUT_DIR=/cluster/scratch/<username>/cil_trainings`.

### 3. Loading environment

1. `cd ./cil-road-segmentation/`
2. Load the **leonhard** software modules: `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
3. Load the python environment: `source ./cil_venv/bin/activate`
4. If you want to work with tmux, start tmux with `tmux`

### 4. Run the training

1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the road segmentation folder `cd road_segmentation_main/`
3. Run a training job on the GPU using the python script `train.py`
    - First select a configuration file. All configuration files can be found in the folder `.configurations/`.
    - Example to run a job using the default configuration file `./configurations/default.jsonc`:
         - 4h run: `bsub -n 2 -J "training-job" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python train.py --configuration ./configurations/default.jsonc'`
         - 24h run with larger dataset: `bsub -n 4 -J "long-run" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/default.jsonc'`
    - Check the job status `bbjobs -w`
    - Peek the `stdout` log `bpeek` or `bpeek -f` to continuously read the log
4. The result of the trainings can be found by default (see [2. Add environment variables](#2-add-environment-variables)) in the folder `./trainings`
   - The folders have following naming convention: `<datetime>-<configfile_name>` (see [Training folder structure](#training-folder-structure))


#### Reproducibility

We used following base submission command on the Leonhard cluster which selects enough cpu memory as well as the 2080Ti
GPU.

```
bsub -n 4 -J "description" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python <....>'
```

To get reproducible results we fixed the random seeds of `torch`, `random` and `numpy` at various points in the code.
Additionally, we set `torch.backends.cudnn.deterministic = True` as suggested on the official pytroch reproducibility
page: https://pytorch.org/docs/1.9.0/notes/randomness.html.

The results of the U-Net are reproducible. The GC-DCNN lacks exact reproducibility because the pyramid pooling
module (PPM) uses the pytorch function `F.interpolate`
which is not numerically stable (as in pytorch version 3.8.5). As a result we evaluated how much the validation accuracy
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

### 5. Run the inference

1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the road segmentation folder `cd road_segmentation_main/`
3. Run an inference job on the GPU using the python script `inference.py`
    - The command line argument `--run_folder` of the inference script `inference.py` takes the path to the trainings' folder created during training, for example: `--run_folder ./trainings/<datetime>-<configfile_name>`
    - **Leonhard** command to run an inference job: `bsub -n 1 -J "submission-job" -W 0:05 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/<datetime>-<configfile_name>'`
4. During the inference job a folder called `prediction-<datetime>` is created inside the `run_folder`. This folder will contain the submission file `submission.csv` (see [Training folder structure](#training-folder-structure)).

### 6. Run an ensemble prediction

1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the road segmentation folder `cd road_segmentation_main/`
3. Run an ensemble job on the GPU using the python script `ensemble.py`
    - The ensemble.py script has the argument `--configuration` which takes the path to the "special" ensemble
      configuration file which is different from the normal configuration files.
        - Contrary to the normal configuration files an ensemble configuration file **needs to be adjusted** because it
          contains a list of relative paths to prediction folders.
        - As a starting point the final ensemble configuration file can be
          used: [ensemble-final.json](./road_segmentation_main/configurations/final/ensemble-final.jsonc)
         ```
         {
             "environment": {
                 "name" : "Name of the Run - this is just a default config file",
                 "output_path": "getenv('OUTPUT_DIR')",
                 "log_file" : "logs.txt"
             },
             "dirs_prediction" : ["20210710-115819-gcdcnn_final/prediction-20210711-095823",
                                  "20210710-115820-gcdcnn_plus_final/prediction-20210711-112218",
                                  "20210710-115820-unet_final_plus/prediction-20210711-095738",
                                  "20210710-115821-unet_final/prediction-20210710-235855",
                                  "20210709-163934-unet_exp_baseline/prediction-20210710-093845",
                                  "20210709-155952-gcdcnn_exp_baseline/prediction-20210711-120117"],
             "mode" : "binary",
             "voting_threshold" : 0.5
         }
         ```
        - The main thing that needs to be adjusted is the parameter `"dirs_prediction"`, which is a list of relative
          paths to prediction folders. By default, the paths are relative with respect to the environment
          variable `OUTPUT_DIR` specified as in [2. Add environment variables](#2-add-environment-variables).
    - To run the ensemble prediction with the final ensemble file one can use following **Leonhard** command:
      `bsub -n 1 -J "ensemble" -W 0:05 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python ensemble.py --configuration configurations/final/ensemble-final.jsonc'`
4. The result of the ensemble prediction can be found in the directory where environment variable `OUTPUT_DIR` points
   to.
    - The folder has the following naming convention: `<datetime>-<configfile_name>`
    - The `submission.csv` file can be found directly in this folder.    