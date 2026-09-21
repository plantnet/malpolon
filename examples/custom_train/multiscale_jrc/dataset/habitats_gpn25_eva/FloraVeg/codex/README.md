# Improvements recommendations
## Yes

  - Start with a frozen-backbone linear-probe baseline, then unfreeze only the final 1–2 transformer blocks or use LoRA/adapters. Full fine-tuning of ViT-L on 4,376 images is likely to overfit; DINOv2 features are specifically designed to work well with linear heads, with fine-tuning often giving modest incremental gains. DINOv2 project documentation, DINOv2 paper

  - Use discriminative learning rates: lowest for unfrozen DINO blocks, higher for the classification heads; switch to AdamW, warm-up, cosine decay, gradient clipping, and early stopping based on validation level-3/4 accuracy.

  - Train all hierarchy levels jointly (1, 2, 3, optionally 3_4) rather than only 2 and 3_4. Use auxiliary level-1/2 losses to regularise scarce level-3/4 labels, with a larger weight on the finest requested output.
  
  - Add hierarchical consistency: use EUNIS parent-child mappings to penalise incompatible predictions, or restrict a fine-level prediction to children of high-probability parent classes. This is especially valuable where visual evidence only supports a broad habitat family.

  - Replace or compare the current soft-target cross-entropy with a long-tail multi-label loss. The local training data has 39 level-3/4 labels with five or fewer assignments and eight singleton labels. Distribution-balanced loss is
    designed to account for both class imbalance and label co-occurrence. Distribution-Balanced Loss

  - Add class-aware sampling or effective-number class weights, but validate it carefully: naïve oversampling can overemphasise common co-occurring labels. Treat rare-label performance as a separate target, not merely an overall top-1 gain.

  - Keep a separate “tail-aware” checkpoint criterion: for example, a weighted combination of overall soft top-1 accuracy, macro-F1, and accuracy on labels with fewer than 20 training assignments. Overall accuracy alone will be dominated by the common habitats.

  - Use landscape-appropriate augmentation: random resized crops, modest colour jitter, blur/compression/noise, and horizontal flip. Avoid aggressive rotations or vertical flips unless they are empirically justified; landscape orientation and horizon structure carry habitat information.

  - Compare CLS-token pooling with attention pooling or mean pooling of DINO patch tokens, and use test-time augmentation or multi-crop inference. Fine habitat cues may occupy only a small region, while a CLS-only representation can underweight them.

  - Run repeated spatial/group-aware validation splits and select hyperparameters by mean and variance across seeds. There are 4,376 train sites and 464 test sites, so one 10% split can be noisy, particularly for rare level-3/4 classes.
  
  - Preserve the image-only model as a first-class baseline. GPS is complete for 2,955/4,376 training records (67.5%) and 316/464 test records (68.1%); a model requiring GPS would discard or fail on roughly one-third of samples.
  
  - For image+GPS training, use a missingness-aware fusion model: include a GPS-present indicator, mask absent coordinates, and apply modality dropout on GPS-present samples. Train image-only, GPS-only, and fused paths jointly; this makes the fused model robust to the actual incomplete-modality regime. TIP: incomplete image–tabular multimodal learning, SMIL: severely missing modality learning


  - Report more than the current soft top-1: hierarchical accuracy, macro metrics, per-frequency-bin performance, calibration, confusion matrices grouped by EUNIS parent, and image-only versus GPS-enabled ablations. This will make genuine
    improvements distinguishable from gains caused by class prevalence or spatial leakage.

## Maybe but probably no
  - Add a GPS-noise and missingness stress test: evaluate image-only, fused-with-GPS, randomly GPS-masked, and coordinate-jittered variants. Report the gain only on the 68% of test records with coordinates, plus the all-test-set result.

  - Audit the label/image pairs before optimisation: identify duplicates, watermark artefacts, corrupted images, contradictory multi-label groups, and labels whose evidence is not visually resolvable. For eight singleton level-3/4 labels,
    improving annotation consistency or collecting a few additional images is likely more valuable than tuning the loss.


## No

  - Consider cautious domain-adaptive self-supervised training on the local images before supervised fitting, but compare it against frozen DINO/LoRA baselines and stop if validation drops. With only ~4.4k images, this is an experiment
    rather than a default recommendation.

  - Encode coordinates more richly than raw latitude/longitude: sinusoidal/Fourier features, equal-area projected coordinates, and possibly coarse spatial cells. Regularise strongly and evaluate with spatially separated splits to prevent
    the GPS branch from memorising local sampling geography.


# Refactored EUNIS multi-head pipeline

`train_eunis_multihead.py` is a standalone, offline refactor of the original multi-head baseline. It predicts any selected combination of EUNIS levels 1, 2, 3, 4, and `3_4`, uses the original multi-hot/soft-label objective and metrics, writes checkpoints/metrics/confusion matrices/predictions, and can optionally fuse `lat`/`lon` with image features.

Configure the run in [`config.yaml`](config.yaml), then run from the repository root (with the project's `sjrc` environment activated if needed):

```bash
python codex/train_eunis_multihead.py
```

Set `data.use_gps: true` to use the `lat` and `lon` columns. To train the requested three heads, set `training.levels: ["1", "2", "3"]` and supply three matching `training.level_weights`. The default metadata paths reproduce the original S3 extended split. Every generated result is placed under `codex/outputs/` by default.

The pipeline deliberately does not download pretrained torchvision weights, complying with offline execution. The original local PN22M checkpoint remains supported by selecting `--model dinov2_PN22M` after installing/importing its local dependencies.
