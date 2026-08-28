# Chest X-ray sample provenance

Complete this file before committing or sharing any demonstration image. Use the exact terminology and label from the source dataset; do not infer a diagnosis from visual inspection.

## Pre-commit checklist

- [ ] Redistribution is permitted by the dataset licence or competition rules.
- [ ] The source dataset and original image identifier are recorded.
- [ ] The original label was copied from the authoritative annotation file.
- [ ] Patient names, identifiers, dates and other identifying overlays are absent.
- [ ] Embedded metadata has been inspected and stripped where necessary.
- [ ] Any conversion, crop, resize or redaction is documented below.
- [ ] The final repository file was reviewed after processing.

## `pneumonia-compatible-opacity.png`

- Intended demo category: `PNEUMONIA_COMPATIBLE_OPACITY`
- Source dataset: `rsna-pneumonia-detection-challenge`
- Dataset or competition URL: `https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge`
- Licence or redistribution terms: `https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules#7-competition-data`
- Exact label: `Lung Opacity`
- Date retrieved: `n/a`
- Original format: `DICOM`
- Processing performed: `original`
- De-identification verified by: `n/a`
- Verification date: `n/a`

## `no-pneumonia-compatible-opacity-01.png`

- Intended demo category: `NO_PNEUMONIA_COMPATIBLE_OPACITY`
- Source dataset: `rsna-pneumonia-detection-challenge`
- Dataset or competition URL: `https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge`
- Licence or redistribution terms: `https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules#7-competition-data`
- Exact label: `No Lung Opacity/Not Normal`
- Date retrieved: `n/a`
- Original format: `DICOM`
- Processing performed: `original`
- De-identification verified by: `n/a`
- Verification date: `n/a`

## `no-pneumonia-compatible-opacity-02.png`

- Intended demo category: `NO_PNEUMONIA_COMPATIBLE_OPACITY`
- Source dataset: `rsna-pneumonia-detection-challenge`
- Dataset or competition URL: `https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge`
- Licence or redistribution terms: `https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules#7-competition-data`
- Exact label: `Normal`
- Date retrieved: `n/a`
- Original format: `DICOM`
- Processing performed: `original`
- De-identification verified by: `n/a`
- Verification date: `n/a`

## Label interpretation

The positive category refers to the source dataset's radiographic finding or annotation compatible with pneumonia. It must not be described as a confirmed clinical diagnosis. The negative category means that the corresponding positive annotation is absent according to the source dataset; it does not certify that the patient has no other abnormality.
