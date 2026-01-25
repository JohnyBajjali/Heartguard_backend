# HeartGuardBE/admin.py
from django.contrib import admin
from django.contrib.auth.models import Group, User
from rest_framework.authtoken.models import Token

from .models import Patient, ECGUpload


# ------------------------------------------------------------------
# ECG UPLOADS (GLOBAL ECG IMAGE DATABASE)
# ------------------------------------------------------------------
@admin.register(ECGUpload)
class ECGUploadAdmin(admin.ModelAdmin):
    list_display = ("id", "file", "source", "uploaded_at")
    list_filter = ("source", "uploaded_at")
    search_fields = ("file",)
    ordering = ("-uploaded_at",)


# ------------------------------------------------------------------
# PATIENTS (MOBILE LOGIN PATIENTS)
# ------------------------------------------------------------------
@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("patient_id", "full_name", "created_at")
    search_fields = ("patient_id", "full_name")
    ordering = ("-created_at",)


# ------------------------------------------------------------------
# HIDE UNWANTED DJANGO MODELS
# ------------------------------------------------------------------
models_to_hide = [Group, User, Token]

for model in models_to_hide:
    try:
        admin.site.unregister(model)
    except admin.sites.NotRegistered:
        pass


# ------------------------------------------------------------------
# ADMIN SITE BRANDING
# ------------------------------------------------------------------
admin.site.site_header = "HeartGuard Admin"
admin.site.site_title = "HeartGuard ECG Dashboard"
admin.site.index_title = "HeartGuard Control Panel"
