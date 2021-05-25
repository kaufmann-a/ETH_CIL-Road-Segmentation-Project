import os
import subprocess
import sys

if __name__ == '__main__':

    directory = "configurations/augmentations"
    GPU_2080TI = False
    TIME = '06:00'
    DEBUG = False

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_path = os.path.join(directory, filename)
        if filename.endswith(".jsonc"):
            command = ''
            if DEBUG:
                command += 'echo '  # test output

            # use 4 cpus
            command += 'bsub -n 4 -J "' + filename[:-6] + '"'
            # job time
            command += ' -W ' + TIME
            # memory per cpu and select one gpu
            command += ' -R "rusage[mem=10240, ngpus_excl_p=1]"'
            if GPU_2080TI:
                command += ' -R "select[gpu_model0==GeForceRTX2080Ti]"'
            else:
                command += ' -R "select[gpu_mtotal0>=10240]"'  # GPU memory more then 10GB

            command += " 'python train.py --configuration " + file_path + "'"

            # new method
            # process = subprocess.run(command.split(), stdout=sys.stdout, stderr=sys.stderr)

            # old method but working
            os.system(command)
