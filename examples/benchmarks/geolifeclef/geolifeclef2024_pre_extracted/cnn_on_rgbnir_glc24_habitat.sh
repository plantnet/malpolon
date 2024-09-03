#!/usr/bin/sh
#OAR -l /nodes=1/gpunum=4,walltime=48:00:00
#OAR -p gpu='YES' and host='nefgpu27.inria.fr'
#OAR --name glc24_MME_habitats_multiclass
#OAR -q dedicated
echo "Launching job $OAR_JOBID on `oarprint gpunb` gpus on host `oarprint host`"

# module purge
# module load conda/2021.11-python3.9
# conda deactivate
source activate malpolon_3.10;

core="python glc24_cnn_multimodal_ensemble_habitat.py"
echo "$core";
$core;
