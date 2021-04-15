# cil-road-segmentation

Software Versions used for this Project (Proposal by Andreas):

- IDE: Pycharm Professional, Version 2020.3.4
- Python-Env: virtualenv
- Python: Version 3.9.2
- Anaconda3: 2020.11 Build

## Setup on leonhard cluster
1. Clone `git clone git@github.com:FrederikeLuebeck/cil-road-segmentation.git`
2. Setup environment
   - Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5`
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
2. Load software modules: `module load gcc/6.3.0 python_gpu/3.8.5`
3. Load python env: `source ./my_venv/bin/activate`

## Run training
1. Load environment
2. Navigate to the road segmentation folder `cd road_segmentation_main/` 
3. Edit the configuration file to your needs
    - `vim ./configurations/default.jsonc`
4. Run job on GPU
   - `bsub -n 1 -J "training-job" -W 4:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" 'python train.py --configuration ./configurations/default.jsonc'`
        - change the configuration file name if you use a different one `--configuration ./configurations/default.jsonc`
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