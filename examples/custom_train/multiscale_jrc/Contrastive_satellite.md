# Run 1
- Name: Contrastive_satellite_gps_try1
- Job_id: 14206398
- Date: 06/05/25

## Analysis
The run gave unsatisfatory results: possible over-fitting.
Train loss & accuracy are through the roof.
Val loss & accuracy is shit.

## Debug
### Sanity check
#### Symetric features
Inputting features of the same image, with equal transformation.

Training on a random sample of 5000 GLC24 PA obs, 10 epochs: 
- Train: Loss & metrics converging OK
- Val: Loss & metrics converging OK