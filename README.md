# CIL Course Project 2021: Road Segmentation

## Abstract

Recent research in the field of semantic image segmentation predominantly directs attention to the development of slight architecture variations with increasing complexity. In this work, we analyze the impact of architecture modifications of a U-Net, namely the GC-DCNN and other self-developed variations.  Our experiments with fine tuning and model architecture alterations lead us to a novel better variant of  GC-DCNN. We also propose two novel post-processing techniques to remove artefacts in predictions. Although these methods improve the visual quality, they do not improve prediction accuracy significantly due to patch abstraction. With the help of numerous experiments, we conclude that the greatest improvement is observed by making the training dataset diverse.

## Index
- [Findings and results](#findings-and-results)
- [Project Code Structure](#project-code-structure)
- [Training Folder Structure](#training-folder-structure)
- [Reproducibility](#reproducibility)

## Findings and Results

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
  , [GMaps-custom](./data/training/gmaps_custom) (The training set downloaders can be found in: [./other/maps](./other/maps))

### Models

We employed two models the [U-Net](http://arxiv.org/abs/1505.04597) and
the [GC-DCNN](https://www.sciencedirect.com/science/article/pii/S0020025520304862). To evaluate the influence of the
architecture we additionally adapt both models to improve the predictive results. In the following we refer to the
**U-Net-Plus** as the U-Net where we increased the pool kernel size from 2 to 4 resulting in a slight improvement.
The **GC-DCNN-Plus** refers to a deeper version of the original GC-DCNN and can be viewed as a novel combination of the
GC-DCNN with the modules [Atrous Spatial Pyramid Pooling](https://arxiv.org/abs/1606.00915v2) (used as a bridge
replacing the Pyramid Pooling Module) and the [attention gate](https://arxiv.org/abs/1804.03999v3) (used in the upwards
branch).
#### Implementation
The implementation of the models can be found
under [./road_segmentation_main/source/models](./road_segmentation_main/source/models). The GC-DCNN implementation is
based on the official paper. From the paper it is not clear at which positions batch normalization is employed. We found
that using batch normalization after almost every convolution layer helps that the model does not diverge during
training. To simplify architecture changes of the U-Net and the GC-DCNN we implemented the models in such a way that we
could change the architecture in the configuration file (e.g. GC-DCNN
implementation: [gcdcnn_bn.py](./road_segmentation_main/source/models/gcdcnn_bn.py)). For instance the model
configuration of the **GC-DCNN-Plus** looks as following in the configuration file:

```
  "model": {
      "name": "gcdcnn_bn",
      "features" : [64, 128, 256, 512, 1024],
      "bridge": {
          "use_aspp" : true,
          "aspp_avg_pooling" : true,
          "ppm_bins" : [1, 2, 3, 6]
      },
      "use_attention" : true,
      "upsample_bilinear" : false,
  },
```

### Postprocessing

We implemented different postprocessing techniques to make the quality of predictions better.

1. Classical methods:
    - Repeated dilation followed by same number of erosions
    - Median filtering The filter size and the type was too dependent on the kind of image, so instead of hand tuning
      these we looked for machine learning based solutions.
2. Retrain on binary:
   We used the best predictions of the U-Net & GCDCNN as a training set and used it to retrain the network to learn to
   connect roads by joining lines and remove noisy predictions.
    - U-Net with partial convolution: We replace the normal convolutions with partial convolution layers in UNET. This
      gave sharper and denoised predictions compared to normal UNET.
    - Increasing the receptive field: We tried experiments with increasing dilation which improved connectivity of
      disjoint segments.
3. Learning hough transforms:
   We nudge the network towards predicting connected roads by explicitly presenting possible connected line fragments.

### Results

- the largest factor was contributed by using more data
- the model architecture as well as the postprocessing played an important but in comparison a minor factor


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
       +-- ensemble.py           [script that given mutliple preditions creats an ensemble prediction]
       +-- ...
```

Our code is build such that it allows to

1. Reproduce runs
2. Compare runs
3. Keep results of completed runs

We use configuration files not only to run different models with different configurations but also to reproduce past
runs. Configuration files can be found in the folder `cil-road-segmentation/road_segmentation_main/configurations`. They
allow to change the dataset, data augmentations, model, model parameters, optimizer, learning rate scheduler, and so on.
Moreover, logging with
`tensorboard` and `comet` gives us the ability to track and compare results of different runs at ease. For every run a
"run-folder" is created which takes the name `<datetime>-<config-file-name>`. This folder keeps the `stdout` log,
the `tensorboard` log and additionally the model weights-checkpoint (see [Training folder structure](#training-folder-structure). 
This folder serves as a back up of executed runs.

`train.py`:
This is the main script to run a training. The main commandline argument is `--configuration` which takes the path to
the configuration file.

`inference.py`: This script helps to get model predictions using the ETH test dataset. The main commandline argument
is `--run_folder` which takes the path to the "run-folder" created during training. Then this script will automatically
load the best model checkpoint and create the submission.csv file inside the "run-folder" in the
folder `prediction-<datetime>`.

### Training folder structure

```
+-- trainings
    +-- <datetime>-<config-file-name>     [this is a training "run-folder"]
        +-- prediction-<datetime>         [contains the model predictions and the submission file]
        |   +-- <config-file>             [copy of the configuration file used in inference]
        |   +-- submission.csv            [the submission file to hand in predictions]
        +-- tensorboard                   [contains the tensorboard log]
        |   +-- events.out.tfevents.*
        +-- weights_checkpoint            [contains the model checkpoints]
        |   +-- <epoch>_*.pth             [model checkpoint file]
        |   +-- ...
        +-- <config-file>                 [copy of the configuration file used in training]
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

#### 1.1 Clone repository and setup environment

1. Clone this git repository `git clone https://github.com/FrederikeLuebeck/cil-road-segmentation.git`
2. Environment setup
    - Load the leonhard software modules:  `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
    - Create a virtual environment and install the required python packages:
        - `cd ./cil-road-segmentation/`
        - `python -m venv cil_venv`
        - `source ./cil_venv/bin/activate`
        - `pip install -r ./road_segmentation_main/requirements.txt`

#### 1.2 Download the data

1. Go to https://polybox.ethz.ch/index.php/s/la14vk4qmlCdRof and download the zip folder `data.zip`
2. Unzip the folder directly into the root folder `cil-road-segmentation`
3. The directory structure should now be as following:
   ```
   +-- cil-road-segmentation
      +-- data
          +-- binary_test_images
          +-- ...
          +-- test_images
          +-- training
      +-- ...
   ```

### 2. Add environment variables

1. Create a file called `.env` in the folder `cil-road-segmentation/road_segmentation_main`. This file should contain
   the configuration of the data directory as well as the output directory.
    - `cd road_segmentation_main/`
    - `vim .env`
2. Add the following environment variables to the file:
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
        - 4h
          run: `bsub -n 2 -J "training-job" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python train.py --configuration ./configurations/default.jsonc'`
        - 24h run with larger
          dataset: `bsub -n 4 -J "long-run" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/default.jsonc'`
    - Check the job status `bbjobs -w`
    - Peek the `stdout` log `bpeek` or `bpeek -f` to continuously read the log
4. The result of the trainings can be found by default (see [2. Add environment variables](#2-add-environment-variables)) in the folder `./trainings`
    - The folders have following naming convention: `<datetime>-<config-file-name>` (see [Training folder structure](#training-folder-structure))

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

##### Ensemble prediction

For the ensemble prediction we combined the results of all five runs listed above in [Baselines](#baselines)
and [Final](#final). To execute the ensemble prediction follow the steps listed
in [6. Run an ensemble prediction](#6-run-an-ensemble-prediction).

##### Postprocessing: binary retraining

We applied the postprocessing on the runs U-Net+, GC-DCNN+ and the ensemble prediction.
In [7. Postprocessing using retraining](#7-postprocessing-using-retraining) we show how these results can be reproduced.

##### Intermediate results

The commands to reproduce the intermediate experiments can be found
in: [intermediate_experiments.md](./intermediate_experiments.md)

### 5. Run the inference

1. Load the environment ([3. Loading environment](#3-loading-environment))
2. Navigate to the road segmentation folder `cd road_segmentation_main/`
3. Run an inference job on the GPU using the python script `inference.py`
    - The command line argument `--run_folder` of the inference script `inference.py` takes the path to the trainings'
      folder created during training, for example: `--run_folder ./trainings/<datetime>-<config-file-name>`
    - **Leonhard** command to run an inference
      job: `bsub -n 1 -J "submission-job" -W 0:05 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/<datetime>-<config-file-name>'`
4. During the inference job a folder called `prediction-<datetime>` is created inside the `run_folder`. This folder will
   contain the submission file `submission.csv` (see [Training folder structure](#training-folder-structure)).

### 6. Run an ensemble prediction

1. Before you can run an ensemble prediction, make sure you executed ([5. Run the inference](#5-run-the-inference)) for every training run you want to include into the ensemble prediction
2. Load the environment ([3. Loading environment](#3-loading-environment))
3. Navigate to the road segmentation folder `cd road_segmentation_main/`
4. Run an ensemble job on the GPU using the python script `ensemble.py`
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
5. The result of the ensemble prediction can be found in the directory where the environment variable `OUTPUT_DIR`
   points to.
    - The folder has the following naming convention: `<datetime>-<config-file-name>`
    - The `submission.csv` file can be found directly in this folder.    

### 7. Postprocessing using retraining

1. Create the test image predictions by running the inference script [4. Run the training](#4-run-the-training).
   After the inference the test images can be found in the folder `run_folder/prediction-<datetime>/pred-masks-original`,
   where `run_folder` is the folder that was supplied as a command line argument to the `inference.py` script. These
   images are later needed in the last step to get the postprocessed test images.
2. Create the binary training dataset by running inference on the entire original dataset used for training.
    1. The `inference.py` script can create the binary training dataset.
    2. To get the predictions of the `experiments_dataset` adjust the configuration file
       parameter `data_collection.collection_names` to `"experiments_dataset"` of the configuration file that is located
       in the `run_folder` (folder created during training):
       ```
       "data_collection": {
           "folder": "getenv('DATA_COLLECTION_DIR')",
           "collection_names": ["experiments_dataset"],
       ```
    3. To create the binary training dataset follow [5. Run the inference](#5-run-the-inference) but additionally set the
       commandline argument `--predict_on_train True`.
    4. Then inside the `run_folder` the folder `prediction-<datetime>/pred-masks-original` contains the binary training
       dataset folder:
       ```
       +-- trainings
           +-- <datetime>-<config-file-name>     [this is a training "run-folder"]
               +-- prediction-<datetime>         [contains the model predictions and the submission file]
                   +-- pred-masks-original       [contains the binary training dataset]
                       +-- experiments_dataset   [this is the binary training dataset]
       ```
3. Run the retraining using [4. Run the training](#4-run-the-training).
   1. To get the results of table III of the report the configuration
      file [unet_exp_dilation_5.jsonc](./road_segmentation_main/configurations/experiments/retrain_binary/unet_exp_dilation_5.jsonc)
      was used.
   2. Before retraining, adjust the path to the dataset folder such that it links to the binary training dataset created
      in the previous **step 2**. Do this by adjusting the parameter `data_collection.folder` in the configuration file.
      The parameter `data_collection.folder` should link to the folder that contains the folders with names as specified
      with the parameter `data_collection.collection_names`. For example:
      ```
      "data_collection": { 
          "folder": "./trainings/<datetime>-<config-file-name>/prediction-<datetime>/pred-masks-original",
          "collection_names": [
              "experiments_dataset"
      ],
      ```
   3. Additionally adjust the path to the test images such that it points to the test image predictions created in **step 1**.
      For that the parameter `data_collection.test_images_folder` of the configuration file needs to be
      adjusted such that it points to the parent folder of the test image predictions created in **step 1**.
      ```
      "data_collection": {
          ...,
          "test_images_folder": "./trainings/<datetime>-<config-file-name>/prediction-<datetime>/pred-masks-original",
      ```
      Attention: This is a different `prediction-<datetime>` folder then that one set for `data_collection.folder`!
   4. The command to run the retraining is: `bsub -n 4 -J "unet_final_plus" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/retrain_binary/unet_exp_dilation_5.jsonc'`
5. Get the final postprocessed predictions of the test image predictions of **step 1** by
   following [5. Run the inference](#5-run-the-inference).

The configuration files used for postprocessing experiments in Tables VI, VII of the report are in the folder [retrain-binary](./road_segmentation_main/configurations/experiments/retrain_binary/).


## Authors
    - Frederike Luebeck
    - Akanksha Baranwal
    - Jona Braun
    - Andreas Kaufmann
