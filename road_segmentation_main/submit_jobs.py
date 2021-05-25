import os
import subprocess

if __name__ == '__main__':

    directory = "configurations/augmentations"
    GPU_2080TI = False
    DEBUG = False

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_path = os.path.join(directory, filename)
        if filename.endswith(".jsonc"):
            command = ''
            if DEBUG:
                command += 'echo '  # test output

            # use 4 cpus
            command += 'bsub -n 4 -J ' + filename[:-6]
            # job time
            command += ' -W 24:00'
            # memory per cpu and select one gpu
            command += ' -R "rusage[mem=10240, ngpus_excl_p=1]"'
            if GPU_2080TI:
                command += ' -R "select[gpu_model0==GeForceRTX2080Ti]"'
            else:
                command += ' -R "select[gpu_mtotal0>=10240]"'  # GPU memory more then 10GB

            command += ' python train.py --configuration ' + file_path

            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            print(output)
