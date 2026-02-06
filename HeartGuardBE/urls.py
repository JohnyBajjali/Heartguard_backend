# HeartGuardBE/urls.py
from django.urls import path

from .views import (
    # Patient
    patient_login_or_create,
    patients_list_create,
    patient_delete,

    # Doctor Notes
    patient_notes,
    patient_note_detail,

    # ECG
    ecg_upload,
    ecg_auto_crop,
    predict_ecg,
    list_ecgs_for_patient,
)

urlpatterns = [
    # -------------------------
    # Patient APIs
    # -------------------------
    path("patient/login_or_create/", patient_login_or_create, name="patient_login_or_create"),
    path("patients/", patients_list_create, name="patients_list_create"),
    path("patients/delete/", patient_delete, name="patient_delete"),

    # -------------------------
    # Doctor Notes APIs
    # -------------------------
    path("patients/<str:patient_id>/notes/", patient_notes, name="patient_notes"),
    path("patients/<str:patient_id>/notes/<int:note_id>/", patient_note_detail, name="patient_note_detail"),

    # -------------------------
    # ECG APIs
    # -------------------------
    path("ecg/upload/", ecg_upload, name="ecg_upload"),
    path("ecg/auto_crop/", ecg_auto_crop, name="ecg_auto_crop"),
    path("ecg/predict/", predict_ecg, name="predict_ecg"),
    path("ecg/list/", list_ecgs_for_patient, name="ecg_list_for_patient"),
]