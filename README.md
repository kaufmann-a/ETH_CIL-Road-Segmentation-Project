# cil-road-segmentation

Software Versions used for this Project (Proposal by Andreas):

- IDE: Pycharm Professional, Version 2020.3.4
- Python-Env: virtualenv
- Python: Version 3.9.2
- Anaconda3: 2020.11 Build

## Setup on leonhard cluster
1. Clone `git clone git@github.com:FrederikeLuebeck/cil-road-segmentation.git`
2. Setup environment
   - Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
   - Create virtual env and install packages:   
        - `cd ./cil-road-segmentation/`
        - `python -m venv my_venv`
        - `source ./my_venv/bin/activate`
        - `pip install -r ./road_segmentation_main/requirements.txt`
3. Preprocess images
   - `cd ./cil-road-segmentation/preprocessing/`
   - `python augmentations.py`

## General: loading environment
1. `cd ./cil-road-segmentation/`
2. Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5 tmux/2.6 eth_proxy`
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
   - longer run with larger dataset: `bsub -n 4 -J "long-run" -W 24:00 -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" 'python train.py --configuration ./configurations/default.jsonc'`
   - check job status `bbjobs` 
   - peek stdout log `bpeek`
5. Find your training results with `ls ./trainings/`

## Run submission
1. Load environment
2. Navigate to the road segmentation folder `cd road_segmentation_main/` 
3. Run job on GPU
   - `bsub -n 1 -J "submission-job" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python inference.py --run_folder ./trainings/20210415-181009-default'`
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

### Transformations

- To transform the original images, run the script `augmentations.py`
 
### Additional Training data

After downloading the folders from OneDrive, rename them as follows: "github_ale", "github_mat", "github_jkf" and rename the folders within these: "images" and "masks".
There are two possible approaches for including them:
1. Train only on the additional data, then use these pretrained weights (change param "main_folder_name" and "transform_folders")
2. Train on our data + additional data together, then use the additional folder in "additional_training_folders". For this, all images need to be of the same size. "github_jkf" is 608x608, so you can run the script preprocessing/crop_images.py
``