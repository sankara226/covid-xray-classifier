# COVID-19 Chest X-Ray Classifier: accuracy, leakage and shortcut learning

A binary image classifier (COVID-19 vs Normal) built with transfer learning, plus an audit that checks whether the model really looks at the lungs.

> **Educational project. Not a medical device and not a diagnostic tool.** Do not use it for any health decision.

The most useful result is not the score. It is the finding that **a strong score on this public dataset is partly explained by where the images come from**, and the experiments that show it.

## Results

Test set: 450 images (225 COVID, 225 Normal), split once and opened once.

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.9434 |
| Test PR-AUC | 0.9525 |
| Decision threshold (chosen on validation, target sensitivity 95%) | 0.1515 |
| Test sensitivity (COVID caught) | 0.9067 (204 of 225) |
| Test specificity (Normal cleared) | 0.7778 (175 of 225) |
| Confusion matrix | TP 204, FP 50, TN 175, FN 21 |

Models compared on validation ROC-AUC:

| Model | Validation ROC-AUC |
|---|---|
| Small CNN from scratch | 0.7225 |
| MobileNetV2, frozen base | 0.9528 |
| MobileNetV2, last 30 layers fine-tuned | 0.9700 |

The threshold gave 95% sensitivity on validation but 91% on test. With only 450 test images, a gap like this is expected and is a reason not to trust a single number.

### The caveat that matters

| Test image version | ROC-AUC |
|---|---|
| Full image | 0.9434 |
| Lungs hidden (centre box masked) | 0.8012 |
| Lungs only | 0.6405 |
| Corner hidden (same-size control) | 0.9123 |

With the lungs hidden, the model still scores 0.80, far above random guessing (0.5). It is using information outside the lungs. The dataset metadata explains why this is plausible: COVID images come from BIMCV, GitHub collections, Eurorad and SIRM, while Normal images come only from two Kaggle sources (RSNA and a pediatric pneumonia set). **No source is shared between the classes**, so class and image source are confounded. Hiding the lungs costs 0.14 AUC while hiding a same-size corner costs 0.03, so the lungs do carry signal, but part of the performance is probably shortcut learning. The masking test is a warning sign, not proof, because the model never saw masked images during training.

Because the test set is balanced, precision is also optimistic. At 1% real prevalence the same model would have a precision of about 4%.

## What this project demonstrates

- A leakage-aware workflow: duplicate removal by file hash before splitting, stratified train/validation/test split, assertions that no file crosses splits, threshold chosen on validation only, test set used once.
- Transfer learning (MobileNetV2, ImageNet weights): frozen training, then fine-tuning with a much smaller learning rate.
- Evaluation beyond accuracy: ROC-AUC, PR-AUC, sensitivity, specificity, confusion matrix, precision at realistic prevalence.
- Model auditing: region masking with a control, Grad-CAM heatmaps, error gallery, source-per-class analysis.

## Project layout

```
configs/default.yaml     all hyperparameters and paths
src/covid_xray/          data, pipeline, model, training, evaluation, audit, Grad-CAM
tests/                   unit tests on synthetic images (no dataset needed)
notebooks/               the original exploration notebook
reports/figures/         generated evaluation figures
docs/MODEL_CARD.md       intended use, data, results, limitations
```

## Quick start

Requires Python 3.10+. A GPU is optional: everything runs on CPU, only slower.

```bash
git clone https://github.com/sankara226/covid-xray-classifier.git
cd covid-xray-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

python -m covid_xray prepare      # download data, deduplicate, split
python -m covid_xray train        # small CNN baseline, then MobileNetV2 transfer learning
python -m covid_xray evaluate     # threshold on validation, report on test
python -m covid_xray audit        # masking experiment and source analysis
python -m covid_xray gradcam      # heatmaps and error gallery
```

Data comes from Kaggle through `kagglehub`. You may need Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`). To try the pipeline quickly, add `--max-per-class 50`.

Run the tests with `make test` and the linter with `make lint`.

The `--max-per-class 50` option is available on every command. To use a local Kaggle dataset copy, pass `--data-dir /path/to/COVID-19_Radiography_Dataset` to `prepare`.

Figures are generated from real model outputs by `evaluate`, `audit`, and `gradcam` in `reports/figures/`. None are committed because this checkout has no dataset or trained model. No metrics or figures were recreated from prose.

## Data

[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Only the `COVID` (3,616) and `Normal` (10,192) folders are used. After removing 47 exact duplicates, 13,761 images remain. For speed, a balanced subsample of 1,500 images per class is split 70/15/15 (2,100 / 450 / 450). Images are 299x299 grayscale and are resized to 224x224 RGB.

The data is not included in this repository.

The original exploration notebook is archived at `notebooks/covid-19.ipynb`. Its saved cell outputs were removed; the project commands above are the reproducible entry point.

## Limitations

- Class and image source are confounded (see above), so results will probably not transfer to another hospital or scanner.
- No patient identifiers: images of the same patient may appear in several splits.
- The Normal class includes pediatric images, so age may act as a shortcut.
- Balanced test set: real-world precision would be much lower.
- One random split of 450 test images and no external validation.

## Roadmap

- Train on masked or lung-segmented images using the dataset's lung masks, and compare.
- External validation: train on some sources, test on others.
- A second project on tabular medical risk prediction, focused on target leakage.

## Credits

Dataset: M.E.H. Chowdhury et al., "Can AI help in screening Viral and COVID-19 pneumonia?", IEEE Access, 2020; T. Rahman et al., "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images", 2021. Pretrained weights: MobileNetV2 on ImageNet, via Keras.

## License

MIT, see [LICENSE](LICENSE).
