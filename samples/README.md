# ED²A demonstration samples

This directory contains five reproducible requests for demonstrating the ED²A web interface and REST API. Each case uses fixed, synthetic demographics and evidence identifiers from the bundled DDXPlus catalogue.

The examples are educational regression fixtures. They are not patient records, medical advice or proof that a diagnosis is present.

## Contents

```text
samples/
├── README.md
├── cases/
│   ├── 01_upper_respiratory.json
│   ├── 02_respiratory_with_xray.json
│   ├── 03_exertional_cardiopulmonary.json
│   ├── 04_sinonasal_allergic.json
│   └── 05_pleuritic_red_flag.json
└── chest-xrays/
    ├── pneumonia-compatible-opacity.png
    ├── no-pneumonia-compatible-opacity-01.png
    ├── no-pneumonia-compatible-opacity-02.png
    └── SOURCES.md
```

The application does not import the JSON files automatically. Enter the same values in the web form or submit a file to `POST /predict-topk`.

## Verified outcomes

The following results were produced with the XGBoost model, preprocessor and label encoder distributed with this project. The X-ray state was checked against the current trigger logic in `templates/index.html`.

| Case | Fixed input | Expected top-ranked diagnosis | X-ray panel |
| --- | --- | --- | --- |
| `01_upper_respiratory.json` | 28, F | URTI | Hidden |
| `02_respiratory_with_xray.json` | 65, F | Acute COPD exacerbation / infection | Shown |
| `03_exertional_cardiopulmonary.json` | 64, M | Stable angina | Hidden |
| `04_sinonasal_allergic.json` | 34, F | Allergic sinusitis | Shown |
| `05_pleuritic_red_flag.json` | 47, M | Pulmonary embolism | Shown |

These expected rankings are specific to the included artifacts. Retraining the model, replacing an artifact or changing a payload can alter them; re-run the cases after any such change.

## Case definitions

### 01 — Upper-respiratory presentation

Expected output: **URTI**. The X-ray panel remains hidden.

- `E_181`: nasal congestion or clear rhinorrhea;
- `E_201`: cough;
- `E_97`: sore throat — selected as the initial evidence;
- `E_48`: living with four or more people;
- `E_222`: daily exposure to second-hand cigarette smoke.

This case provides a straightforward upper-respiratory demonstration without invoking the imaging workflow.

### 02 — Acute respiratory presentation with X-ray workflow

Expected output: **Acute COPD exacerbation / infection**. The X-ray panel is shown.

- `E_91`: fever;
- `E_201`: cough;
- `E_77`: coloured or increased sputum production;
- `E_94`: chills or shivers — selected as the initial evidence;
- `E_79`: current smoking.

This payload uses `k=10`. With the current clinical artifacts, **Pneumonia is ranked seventh**, so it remains visible before and after image-assisted re-ranking. The panel appears because cough and sputum are combined with systemic evidence such as fever and chills.

The fusion logic was also checked with controlled image-model inputs. With the default threshold and weight, a supporting score of `0.9` increased Pneumonia from `2.507%` to `4.475%` and moved it from rank 7 to rank 6. A non-supporting score of `0.1` reduced it to `1.391%`. These values verify the fusion algorithm; they are not outputs produced from the radiographs stored in this directory.

### 03 — Exertional cardiovascular presentation

Expected output: **Stable angina**. The X-ray panel remains hidden.

- `E_218`: symptoms worsen with exertion and improve with rest — selected as the initial evidence;
- `E_89`: persistent fatigue or non-restorative sleep;
- `E_79`: current smoking;
- `E_71`: hypercholesterolaemia or lipid-lowering therapy;
- `E_225`: cardiovascular disease in a close relative before age 50.

The fixed payload is intended to contrast a cardiovascular differential with the respiratory cases.

### 04 — Sinonasal allergic presentation

Expected output: **Allergic sinusitis**. The X-ray panel is shown.

- `E_181`: nasal congestion or clear rhinorrhea — selected as the initial evidence;
- `E_201`: cough;
- `E_226`: predisposition to common allergies;
- `E_124`: asthma or previous bronchodilator use;
- `E_169`: itching of the nose or back of the throat.

The panel appears through the respiratory-plus-context rule: cough (`E_201`) is combined with asthma or prior bronchodilator use (`E_124`).

### 05 — Pleuritic red-flag presentation

Expected output: **Pulmonary embolism**. The X-ray panel is shown.

- `E_151`: swelling in one or more body areas;
- `E_220`: pain worsened by deep inspiration — selected as the initial evidence;
- `E_66`: significant shortness of breath;
- `E_109`: previous deep-vein thrombosis;
- `E_196`: surgery within the previous month.

`E_220` is configured as a strong single trigger for the imaging workflow. The complete case also supplies symptoms and thromboembolic risk factors that are coherent with the expected differential.

## Validation scope

All five cases were checked as follows:

1. parsed through the current `TopKPredictionRequest` schema;
2. verified against `release_evidences.json`;
3. executed through the shipped preprocessor and XGBoost model;
4. compared with the expected top-ranked diagnosis in the table above;
5. evaluated against the current frontend X-ray trigger rules;
6. for case 02, checked that Pneumonia remains in the clinical Top-10 and that controlled positive and negative image scores modify its integrated value in the expected direction.

The checks cover deterministic software behaviour for the included artifacts. They do not establish clinical validity, diagnostic accuracy on these synthetic cases or generalisability to real patients.

## Chest X-ray examples

| File | Demonstration use |
| --- | --- |
| `pneumonia-compatible-opacity.png` | Exercise the image-model path with an image carrying the source dataset's positive label. |
| `no-pneumonia-compatible-opacity-01.png` | Exercise the same path with the first source-labelled negative image. |
| `no-pneumonia-compatible-opacity-02.png` | Repeat the negative-label demonstration with a second image. |

Labels must be copied from the authoritative dataset annotations, never assigned by visual inspection. Before committing an image, complete `chest-xrays/SOURCES.md` with its provenance, original label, licence, transformations and de-identification checks.

Every uploaded image must meet the application contract:

- JPEG, PNG or WebP;
- no larger than 10 MB;
- no DICOM input;
- no patient name, identifier, date of birth or other identifying overlay;
- no identifying information in embedded metadata.

## Running a case

From the repository root:

```bash
curl --request POST 'http://localhost:8000/predict-topk' \
  --header 'Content-Type: application/json' \
  --data-binary '@samples/cases/01_upper_respiratory.json'
```

The integrated endpoint accepts the clinical request as a multipart JSON field and the radiograph as a file:

```bash
curl --request POST 'http://localhost:8000/predict-integrated' \
  --form 'payload={"age":65,"sex":"F","evidences":["E_91","E_201","E_77","E_94","E_79"],"initial_evidence":"E_94","k":10}' \
  --form 'chest_xray=@samples/chest-xrays/pneumonia-compatible-opacity.png;type=image/png'
```

The endpoint always preserves the original clinical ranking in its response. A usable, non-inconclusive image adjusts only the Pneumonia entry in the experimental integrated ranking. Case 02 is configured with `k=10` specifically so that the clinical and integrated Pneumonia values can be compared.

## Suggested demonstration sequence

1. Start ED²A and confirm that the required services are operational in `GET /health`.
2. Run case 01 to show a simple URTI result without image input.
3. Run case 02 to open the imaging workflow, then compare a source-labelled positive and negative radiograph.
4. Run case 03 to show a cardiovascular result with the imaging workflow hidden.
5. Run case 04 to demonstrate the respiratory-plus-context X-ray trigger.
6. Run case 05 to demonstrate a pleuritic red flag and the strong-single-evidence trigger.

For complete API contracts, Docker commands and model limitations, see the repository-level `README.md`.
