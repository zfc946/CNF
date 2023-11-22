'''
Script used to generate the results for the exact normal experiment.
'''
import subprocess
from glob import glob
import re
import os



subprocess.run(['python', 'fit_pointcloud.py', '-s circle', '-n 9',])

log = max(glob('lightning_logs/*/checkpoints/*.ckpt'), key=os.path.getmtime)
checkpoint_version = int(re.split('/', log)[1][8:])
print('using checkpoint version {}'.format(checkpoint_version))


subprocess.run(['python', 'fit_pointcloud.py', '-s line', '-n 9', '-p {}'.format(checkpoint_version)])
subprocess.run(['python', 'fit_pointcloud.py', '-s triangle', '-n 9', '-p {}'.format(checkpoint_version)])


subprocess.run(['python', 'fit_pointcloud.py', '-s circle', '-n 12',])
log = max(glob('lightning_logs/*/checkpoints/*.ckpt'), key=os.path.getmtime)

checkpoint_version = int(re.split('/', log)[1][8:])
print('using checkpoint version {}'.format(checkpoint_version))

subprocess.run(['python', 'fit_pointcloud.py', '-s diamond', '-n 12', '-p {}'.format(checkpoint_version)])
