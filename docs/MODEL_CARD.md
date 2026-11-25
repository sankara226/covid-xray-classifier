# Model card: COVID-19 chest X-ray classifier

**Status:** educational. Not a diagnostic tool. Not validated for any clinical use.

## Model
MobileNetV2 (ImageNet weights) with a new head: global average pooling, dropout 0.3, one sigmoid unit. Input 224x224 RGB. Light augmentation during training (rotation 0.05, zoom 0.1, contrast 0.1, no horizontal flip). Trained in two stages: frozen base (learning rate 1e-3, early stopping on validation AUC), then the last 30 layers fine-tuned at 1e-5.

## Intended use
Learning exercise on image classification, evaluation and data leakage. Not intended for screening, triage or diagnosis.

## Data
COVID-19 Radiography Database (Kaggle), classes COVID and Normal only. 47 exact duplicates removed by MD5 (13,808 to 13,761 images). Balanced subsample of 1,500 per class, stratified split 70/15/15 (2,100 / 450 / 450).

| Class | Sources (images) |
|---|---|
| COVID | bimcv (2,474), github collections (765), eurorad (258), sirm (119) |
| Normal | kaggle only: RSNA (8,851) and a pediatric pneumonia set (1,341) |

## Results (test set, 450 images)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9434 |
| PR-AUC | 0.9525 |
| Threshold (chosen on validation for 95% sensitivity) | 0.1515 |
| Sensitivity | 0.9067 |
| Specificity | 0.7778 |
| TP / FP / TN / FN | 204 / 50 / 175 / 21 |

Precision at other prevalences, from these sensitivity and specificity values: about 4% at 1% prevalence.

## Shortcut audit (test ROC-AUC)

| Version | AUC |
|---|---|
| Full image | 0.9434 |
| Lungs hidden | 0.8012 |
| Lungs only | 0.6405 |
| Corner hidden (control) | 0.9123 |

## Limitations
- Class and image source are confounded: no source appears in both classes. Part of the measured skill probably comes from non-lung cues.
- The Normal class includes pediatric images, so age may act as a shortcut.
- No patient identifiers, so patient-level leakage across splits cannot be excluded.
- Balanced test set, so precision is optimistic compared with real prevalence.
- Single random split, no external validation. The threshold reached 95% sensitivity on validation but 91% on test.
- The masking test is indicative, not conclusive, because the model never saw masked images in training.

## Ethical considerations
Medical imaging models can fail silently on populations, scanners or hospitals that differ from the training data. This model must not be used on real patients.

## Citation
Chowdhury et al., IEEE Access, 2020; Rahman et al., 2021 (dataset). See the README for full references.