# HeartGuard — Django Backend

HeartGuard is a graduation project from **Birzeit University** that combines ECG image digitisation, digital-signal processing, and AI-based cardiac classification with a Flutter application and Django backend.

This repository contains the backend: patient records, ECG uploads, notes, preview generation, prediction storage, and integration with the segmentation and classification models.

## Project Team

- Robert Bannoura
- Mahmoud Awad
- Johny Bajjali

**Supervisor:** Dr. Hanna Bullata  
**Department:** Electrical & Computer Engineering  
**Project report:** February 2026

This is a collaborative project. The repository was copied from the team's original backend repository while preserving Git commit history and contributor attribution. Hosting this copy under Johny Bajjali's account does not imply sole authorship of the code or models.

Johny's contribution focused on the application layer, including Flutter development, Django API integration, database setup, patient records, ECG uploads, and presentation of model results.

## Related Repositories

- [Flutter frontend](https://github.com/JohnyBajjali/Heartgurad_frontend)
- [Original team backend repository](https://github.com/RobertBannoura/heartguard-backend)

Private repositories require access permission.

## System Overview

HeartGuard accepts digital ECG signals or images of ECG pages:

1. **Digital signals:** Philips representative-beat XML and HL7 aECG are parsed into lead signals.
2. **Images:** ECG images pass through preprocessing, nnU-Net segmentation, and waveform digitisation.
3. **Classification:** Preprocessed signals are passed to a prototype-based fusion classifier integrating rhythm, morphology, and global ECG representations.
4. **Application workflow:** Results and available previews can be stored and retrieved through the Flutter client alongside patient history and notes.

Numerical model scores are research outputs, not confirmed clinical diagnoses.

## Features

- Patient creation, listing, lookup, and deletion.
- Upload and storage of ECG images, Philips XML, and aECG files.
- Patient ECG history with prediction metadata and timestamps.
- Patient notes and individual ECG notes, with editing and deletion.
- Automatic image cropping using OpenCV.
- Philips XML preview rendering through `HeartGuardBE/inference/xml_preview.py`.
- aECG plotting and digitised-image preview code paths.
- Integration with nnU-Net and a PyTorch fusion predictor.
- Django administration for patients and uploads.
- JWT token obtain/refresh routes.

The source accepts `3x4` and `3x4+1` image-layout labels. The reviewed active image-prediction helper selects dataset 504 for both; the layout label alone does not establish separate dataset-503 inference.

## Technology Stack

| Layer | Technologies |
|---|---|
| Backend | Python, Django, Django REST Framework, SimpleJWT |
| Database | PostgreSQL, Django ORM and migrations |
| Image and signal processing | OpenCV, NumPy, SciPy, SimpleITK, lxml |
| Visualisation | Matplotlib |
| Model integration | PyTorch, torchvision, pandas, nnU-Net v2 |
| Administration and browser integration | django-jazzmin, django-cors-headers |
| Model storage | Git LFS |

## Repository Structure

```text
DjangoProject/                  Django settings and project URLs
HeartGuardBE/
  models.py                    Patients, uploads, notes and related models
  views.py                     Request handlers and inference orchestration
  urls.py                      Application API routes
  migrations/                  Database schema migrations
  inference/                   Image digitisation and preview utilities
  nnUNet_results/              nnU-Net model metadata and checkpoint paths
  protoecgnet_Model/            Signal parsers, preprocessing and fusion predictor
  resources/                   ECG label metadata
manage.py                      Django command entry point
```

Uploaded files and generated previews/inference artifacts are stored under `media/` at runtime. Patient history is maintained as an ordered list of upload IDs on the patient record.

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/token/` | Obtain JWT tokens |
| POST | `/api/token/refresh/` | Refresh access token |
| POST | `/api/patient/login_or_create/` | Patient lookup/creation using name and ID |
| GET, POST | `/api/patients/` | List/create patients |
| POST | `/api/patients/delete/` | Delete patient |
| GET, POST | `/api/patients/<patient_id>/notes/` | List/add patient notes |
| PATCH, PUT, DELETE | `/api/patients/<patient_id>/notes/<note_id>/` | Edit/delete patient note |
| POST | `/api/ecg/upload/` | Upload ECG file and link it to patient history |
| POST | `/api/ecg/auto_crop/` | Return cropped JPEG image |
| POST | `/api/ecg/predict/` | Parse/process ECG and return model scores |
| GET | `/api/ecg/list/?patient_id=<id>` | Retrieve patient ECG history |
| POST | `/api/ecg/delete/` | Delete ECG upload |
| GET, POST | `/api/ecg/notes/` | List/add ECG notes |
| GET, POST | `/api/ecg/notes/add/` | Alias to the ECG-note handler |
| PATCH, PUT, DELETE | `/api/ecg/notes/<note_id>/` | Edit/delete ECG note |

Uploads and predictions accept multipart form data with a `file` field. Uploads also require `patient_id` and `patient_name`; predictions can receive an existing `ecg_id`. Prediction responses include `topk` label/score entries and format-specific metadata.

The backend fusion predictor is configured for 71 labels and returns up to 10 ranked entries. The Flutter client groups diagnostic entries into NORM, MI, STTC, CD, and HYP using the supplied label CSV.

## Local Setup Requirements

The repository is source code for a research project, not a packaged hosted service. To run it locally, prepare:

- A compatible Python environment with the team's backend dependencies.
- PostgreSQL and a local database/user.
- Git LFS and the actual model checkpoint files.
- nnU-Net v2 with `nnUNetv2_predict` available on the process PATH.
- Appropriate PyTorch/torchvision packages for the available hardware.

### Clone with model files

```powershell
git lfs install
git clone https://github.com/JohnyBajjali/Heartguard_backend.git
cd Heartguard_backend
git lfs pull
```

Git LFS weights were uploaded to this repository during migration. A GitHub ZIP may still contain pointer text instead of actual model weights; use a Git LFS-enabled clone for inference.

### Configure PostgreSQL and Django

The reviewed settings read these environment variables:

```text
DJ_SECRET_KEY
PG_DB
PG_USER
PG_PASSWORD
PG_HOST
PG_PORT
```

Set them in the environment used to start Django. Keep passwords and signing keys local.

The existing `.env.example` uses different variable names, and the reviewed startup code does not load `.env` automatically. Copying that example alone does not configure these settings.

### Prepare dependencies and start Django

No backend dependency manifest was present in the reviewed archive. Recover the requirements from the team's working environment before attempting a reproducible installation; the technology list above is not a version-locked requirements file.

Once dependencies, model files, database, and environment variables are ready:

```powershell
python -m venv .venv
# Install the team's recovered dependencies into .venv before continuing.
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py createsuperuser
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

The local API is served at `http://127.0.0.1:8000/api/`; Django admin is at `/admin/`. Database schema is created through migrations; real patient records are not required and must not be published for setup.

## Project Report

The team report, **“Heart Guard: An AI-Based System for ECG Image Digitisation and Cardiac Diagnosis”**, covers dataset preparation, segmentation, digitisation, prototype-based classification, evaluation, and the application implementation.

Useful sections include Chapter 3 (system architecture), Chapters 5–7 (processing/model methods), and Chapter 8 (Flutter/Django implementation and API mapping). The report describes the overall team project; this README describes the backend repository and its setup requirements.

The report is not bundled by this README. If it is published separately, use a copy reviewed for patient identifiers and third-party material, with the team's agreement.

## Research and Development Status

- This is a research prototype, not a clinically approved diagnostic product.
- Adding the recovered XML preview module and uploading LFS artifacts restores those missing distribution components; a clean end-to-end installation was not retested during repository preparation.
- The reviewed image helper passes PNG data to the nnU-Net runner with `is_png=False`, which changes the filename to `.nii.gz` without converting the content. This integration detail needs verification before claiming that the current checkout's image pipeline is ready to run.
- Current settings enable debug mode, unrestricted allowed hosts, and broad CORS. Patient/ECG routes are plain Django views; DRF JWT defaults do not automatically secure them. Do not expose this development configuration as a public patient-data service.
- Only synthetic or properly de-identified ECGs should be used for public demonstrations.

## Attribution

Original team repository: [RobertBannoura/heartguard-backend](https://github.com/RobertBannoura/heartguard-backend). Contributor history is preserved. Model and third-party research credits are documented in the project report; reuse remains subject to the applicable permissions and terms.
