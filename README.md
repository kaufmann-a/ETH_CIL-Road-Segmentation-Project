# cil-road-segmentation

Software Versions used for this Project (Proposal by Andreas):

- IDE: Pycharm Professional, Version 2020.3.4
- Python-Env: virtualenv
- Python: Version 3.9.2
- Anaconda3: 2020.11 Build

## Setup on leonhard cluster
1. Clone `git clone git@github.com:FrederikeLuebeck/cil-road-segmentation.git`
2. Setup environment
   - Load software modules: `module load gcc/6.3.0 cuda/11.0.3 cudnn/8.0.5 python_gpu/3.8.5 tmux/2.6 eth_proxy`
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
    - On the leonhard cluster it is advisable to use the scratch as output directory,
      due to space constraints of the home directory. For instance use
      `OUTPUT_DIR=/cluster/scratch/<username>/cil_trainings`.

## General: loading environment
1. `cd ./cil-road-segmentation/`
2. Load software modules: `module load gcc/6.3.0 cuda/11.0.3 cudnn/8.0.5 python_gpu/3.8.5 tmux/2.6 eth_proxy`
3. Load python env: `source ./my_venv/bin/activate`
4. If you want to work with tmux, start tmux with `tmux` (see tmux guide below)

## Run training
1. Load environment
2. Navigate to the road segmentation folder `cd road_segmentation_main/` 
3. Edit the configuration file to your needs
    - `vim ./configurations/default.jsonc`
4. Run job on GPU
   - shorter run: `bsub -n 2 -J "training-job" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python train.py --configuration ./configurations/default.jsonc'`
        - change the configuration file name if you use a different one `--configuration ./configurations/default.jsonc`
   - longer run with larger dataset: `bsub -n 4 -J "long-run" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/default.jsonc'`
   - check job status `bbjobs -w` 
   - peek stdout log `bpeek` or `bpeek -f` to continuously read the log
5. Find your training results with `ls ./trainings/`

### Reproducibility

We used following base submission command on the Leonhard cluster which selects enough cpu memory as well as the 2080Ti GPU.
```
bsub -n 4 -J "description" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python <....>'
```

To get reproducible results we fixed the random seeds of `torch`, `random` and `numpy` at various points in the code.
Additionally, we set `torch.backends.cudnn.deterministic = True` as suggested on the official pytroch reproducibility 
page: https://pytorch.org/docs/1.9.0/notes/randomness.html.

The results of the U-Net are 100% reproducible.
The GC-DCNN lacks exact reproducibility because the pyramid pooling module (PPM) uses the pytorch function `F.interpolate`
which is not numerically stable (as in pytroch version 3.8.5). As a result we evaluated how much the validation accuracy varies,
by running 10 runs of the GC-DCNN baseline using the "experiments dataset".
- min validation accuracy: 0.9724 (removed one outlier with validation accuracy of 0.9715)
- max validation accuracy: 0.9732

#### Baselines
For the baselines we use the "experiments dataset" which is a combination of the ETH and the GMaps-public dataset with a 
predefined train and validation split. The images of the GMaps-public dataset were center cropped to size 400x400 to match the
ETH dataset image size.

| Model | Command |
| ----- | ------- |
| U-Net |`bsub -n 4 -J "gcdcnn_exp_baseline" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/gcdcnn_exp_baseline.jsonc'`|
| GC-DCNN |`bsub -n 4 -J "unet_exp_baseline" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/experiments/unet_exp_baseline.jsonc'`|

#### Final
For our final submission we used the datasets: ETH, GMaps-public, GMaps-custom with a validation split of 20%.

| Description | Command |
| ----------- | ------- |
| U-Net<br>Augmentations: SSR, RC|`bsub -n 4 -J "unet_final" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/unet_final.jsonc'`|
| U-Net+<br>Augmentations: SSR, RC|`bsub -n 4 -J "unet_final_plus" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/unet_final_plus.jsonc'`|
|GC-DCNN<br>Augmentations: SSR, RC, GN|`bsub -n 4 -J "gcdcnn_final" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/gcdcnn_final.jsonc'`|
|GC-DCNN+<br>Augmentations: SSR, RC, GN|`bsub -n 4 -J "gcdcnn_final_plus" -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration configurations/final/gcdcnn_final_plus.jsonc'`|

## Run submission
1. Load environment
2. Navigate to the road segmentation folder `cd road_segmentation_main/` 
3. Run job on GPU
   - `bsub -n 1 -J "submission-job" -W 0:05 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/20210415-181009-default'`
        - change the run folder name to the trainings' folder created during training: `--run_folder ./trainings/20210415-181009-default`
4. Find the submission file in your run folder

## Tmux guide
- Enter tmux: `tmux`
- Split panes left/right: `Ctrl+b %`
- Split panes top/bottom: `Ctrl+b "`
- Switch panes: `Ctrl+b ArrowKeys`
- Closing panes: `Ctrl+d` or `exit` (by clossing all panes you exit the session)
- Detach Session (session keeps running in background): `Ctrl+b d`
- Reattach to running session: Check out running sessions with `tmux ls` then attach to session with `tmux attach -t 0` where 0 is session 0.
- Enter scroll-mode (to scroll up and down): `Ctrl+b [` then arrow keys (to quit press `q`)

### Other commands
- New window: `Ctrl+b c`
- Switch windows back and forth: `Ctrl+b p`, `Ctrl+b n` or `Ctrl+b WindowNr`
- Toggle a pane fullscreen and embeded: `Ctrl+b z`

## Preprocessing


 
### Prepare additional Training data on cluster scratch

1. Connect to leonhard and navigate to the scratch `cd /cluster/scartch/...`
2. load a newer version of curl: `module load curl`
3. Open your browser (chrome, firefox) and navigate to oneDrive
4. Open developer tools: `CTRL+SHIFT+I`
5. Navigate to the  "Networks" tab and enter "zip?" in the filter (networkstack might now be empty)
6. Select all folders you like to download and click on "Download"
7. cancel the download
8. Now there should be one entry in the Networks tab
9. Right click on it: Copy -> Copy as cURL
10. Paste the command into your leonhard terminal and add `-o data.zip` to specify the output file
11. Hit enter and wait
12. Unzip the data `unzip data.zip`  (with -d to add target directory path)

### Transformations

1. cd to location of augmentations script `cd /cluster/home/{username}/cil-road-segmentation/preprocessing`
2. Load environment
3. run augmentation.py, add location of data as argument: `python augmentations.py "/cluster/scratch/{username}/data"`
