#!/usr/bin/sh
#OAR -l gpunum=4,walltime=72:00:00
#OAR -p gpu='YES' and host='nefgpu27.inria.fr'
#OAR --name Irokube_no-workers_save300steps
#OAR -q dedicated
echo "Launching job $OAR_JOBID on `oarprint gpunb` gpus on host `oarprint host`"

# module purge
# module load conda/2021.11-python3.9
# conda deactivate
source activate malpolon_3.10;

core="python cnn_on_bioclim_torchgeo.py"
echo "$core";
$core;
