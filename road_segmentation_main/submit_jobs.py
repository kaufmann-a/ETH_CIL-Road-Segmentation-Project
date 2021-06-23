import os
import subprocess
import sys

if __name__ == '__main__':

    directory = "configurations/experiments/unet-experiments/"

    GPU = 2  # None (= more then 10GB GPU memory), 1 (=1080Ti), 2 (=2080Ti)
    TIME = '24:00'
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
            if GPU == 1:
                command += ' -R "select[gpu_model0==GeForceGTX1080Ti]"'
            elif GPU == 2:
                command += ' -R "select[gpu_model0==GeForceRTX2080Ti]"'
            else:
                command += ' -R "select[gpu_mtotal0>=10240]"'  # GPU memory more then 10GB

            command += " 'python train.py --configuration " + file_path + "'"

            print(command)

            # new method
            # process = subprocess.run(command.split(), stdout=sys.stdout, stderr=sys.stderr, shell=True)

            # old method but working
            os.system(command)
