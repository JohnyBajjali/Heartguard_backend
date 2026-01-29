# HeartGuardBE/urls.py
from django.urls import path
from .views import (
    ecg_upload,
    ecg_auto_crop,
    patient_login_or_create,
    list_ecgs_for_patient,
    patients_list_create,
    predict_ecg,
)

urlpatterns = [
    path("patient/login_or_create/", patient_login_or_create, name="patient_login_or_create"),
    path("patients/", patients_list_create, name="patients_list_create"),
    path("ecg/upload/", ecg_upload, name="ecg_upload"),
    path("ecg/auto_crop/", ecg_auto_crop, name="ecg_auto_crop"),
     path("ecg/predict/", predict_ecg, name="predict_ecg"),
    path("ecg/list/", list_ecgs_for_patient, name="ecg_list_for_patient"),
]
