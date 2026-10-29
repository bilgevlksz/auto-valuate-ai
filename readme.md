# Auto Valuate AI

Used-car asking-price prediction experiments by **Bilge Evleksiz**, combining vehicle information with image features extracted using **ResNet18** and **Vision Transformer (ViT)** models.

The project includes a Jupyter notebook with saved regression experiments and a **Streamlit prototype** using ViT image features and a trained CatBoost regressor. This documentation update preserves the existing notebook, application code and model file; no model was retrained and no new performance result was generated.

## Project overview

The goal is to explore price prediction using two types of input:

- **Tabular information:** mileage, brand, series, model, body type, fuel type, transmission and engine displacement.
- **Vehicle photographs:** general visual feature vectors extracted by ImageNet-pretrained networks.

The notebook's saved data-parsing output reports **7,851 listings**. The full source dataset is not included, so the raw/filtered counts and data provenance have not been independently reproduced. These are listing asking prices, not verified transaction prices.

## Method

1. Parse listing text files into a table and associate each listing with its photograph folder.
2. Prepare numeric fields and categorical features.
3. Extract a vector from each of up to ten photos per listing:
   - **ResNet18:** replace `fc` with `Identity` to obtain 512-dimensional features.
   - **ViT-B/16:** replace `heads` with `Identity` to obtain 768-dimensional features.
4. Average photo vectors to produce one visual vector per listing.
5. Concatenate visual and tabular features and train a regression model.
6. Evaluate with MAE, RMSE and R².

The visual backbones are used in evaluation mode without gradient computation; they are **feature extractors, not models fine-tuned on the vehicle dataset**. The notebook explores XGBoost and CatBoost. The final ViT/CatBoost branch uses `log1p(price)` during training and `expm1` to return predictions to the original TL scale.

The final CatBoost input contains **776 features**: mileage, seven categorical fields and 768 visual values. Although year exists in the source table, it is not included in this final feature selection.

## Saved experimental results

The table summarizes existing text outputs in `Araba_fiyat_tahmini.ipynb`. These values were not rerun or independently reproduced during documentation preparation.

| Experiment | Notebook cell (1-based) | R² | MAE (TL) | RMSE (TL) |
| --- | --- | --- | --- | --- |
| ResNet18 + XGBoost | 4 | 0.7141 | 129,143.45 | 444,069.93 |
| ResNet18 + CatBoost | 6 | 0.8870 | 129,594.59 | 203,067.18 |
| ResNet18 + CatBoost, log target | 8 | 0.8806 | 127,192.88 | 208,734.76 |
| Premium subset, ResNet18 + CatBoost, log target | 10 | 0.7084 | 419,184.03 | 688,088.84 |
| ViT + CatBoost, log target | 11 | 0.8887 | 122,462.93 | 201,448.73 |

**Evaluation context:** the evaluation split was also used for early stopping/model selection. These are therefore **validation/model-selection results, not independent held-out test scores**. Several experiments filter price quantiles using the entire dataset before splitting; this influences the evaluation population and can bias the assessment. The premium experiment uses a different subset and split fraction, so its metrics are not directly comparable to the full-data experiments.

**Visual contribution:** image vectors genuinely enter the regressors, but the original notebook does not contain a controlled tabular-only comparison. The results therefore do not establish how much photographs improve prediction. R² is not classification accuracy, and MAE does not provide an individual price confidence interval.

## Repository contents

```text
Araba_fiyat_tahmini.ipynb       Original experiments and saved outputs
app.py                        Streamlit ViT/CatBoost prototype
catboost_vit_log_modeli.cbm    Existing trained CatBoost model file
readme.md                     Project documentation
requirements.txt              Core application/experiment dependencies
.gitignore                    Local data, credentials and artifact exclusions
test_images/                  Existing demonstration listing files
```

The notebook and saved outputs remain the record of the original experiments. The `.cbm` file is already in the repository; its filename alone does not establish an independently verified link to a particular printed score. The documentation update does not replace or retrain it.

## Inspect the project

Open [the notebook](Araba_fiyat_tahmini.ipynb) on GitHub to review the code and saved outputs. **No execution or retraining is necessary to inspect this work.**

The application is an experimental local prototype. This repository does not establish production deployment, measured response-time guarantees or validated damage/condition assessment.

## Optional local application setup

These steps are for someone who wants to run the existing application; they are not required to view the repository or apply this documentation update.

```bash
git clone https://github.com/bilgevlksz/auto-valuate-ai.git
cd auto-valuate-ai
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Run from the repository root: the existing application looks for `catboost_vit_log_modeli.cbm` relative to its working directory. The first ViT load needs internet access to download torchvision's ImageNet weights. The application uses CPU inference. The dependency list is not an exact lockfile from the original training environment; full installation, model loading and end-to-end execution were not verified in this documentation update.

The application takes vehicle information and uses up to the first ten uploaded photos, averages their ViT embeddings, combines them with tabular inputs and predicts a price using CatBoost. Uploads should be valid JPG/JPEG/PNG images of the same vehicle.

## Reproduction notes and limitations

- Full training needs the original local dataset, CSV and feature arrays, which are not bundled. The notebook contains Google Colab/Drive paths that must be adapted to another environment. Some original CatBoost training cells explicitly request GPU execution.
- The original feature arrays are matched by row position/index rather than a stored listing ID/hash manifest. Matching is correct only if the source CSV and feature-cache order remain consistent; cache provenance was not independently verified.
- Corrupt images are silently skipped in parts of the original code; when no image is available, the training loader substitutes a dummy tensor. The application also lacks robust handling of an empty valid-vector list. These behaviors were not changed or runtime-tested in this documentation-only update.
- Repeated physical vehicles, sellers and duplicate/near-duplicate photos across data splits have not been audited. A random listing split alone does not guarantee separation of those groups.
- The image embeddings are general visual representations. No supervised damage detector or explicit cosmetic-condition score was trained or validated. Existing interface wording about damage, condition or a ±3.5% confidence interval is not supported by an accompanying validated method and should not be treated as a measured capability. The application does not implement the Plotly gauge mentioned in its explanatory text.
- Brand/model/category formatting can differ between raw listing text and manual application inputs. Category normalization, broader generalization, temporal price changes and calibration remain future work.

Useful next experiments would be a separate train/validation/test design, ID-based feature matching, a controlled tabular-only baseline, addition/evaluation of year, and vehicle/seller/time-aware evaluation. **These improvements are suggestions, not features already implemented by this update.**

## Data and third-party materials

The notebook parses locally supplied listing text/photos, but does not establish the original provider, collection date, authorization or redistribution rights. Existing demonstration photos and listing text should be reviewed under their source terms before further redistribution. No new dataset, photos, pretrained assets or model files are added by this documentation update.

Torchvision, XGBoost, CatBoost and other dependencies retain their respective terms. The model implementation's license does not automatically grant rights to listing photos or pretraining data. No project-wide open-source license is assigned by this update.
