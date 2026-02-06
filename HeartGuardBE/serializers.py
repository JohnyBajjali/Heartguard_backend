# HeartGuardBE/serializers.py
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Hospital, Patient, EcgExam, DoctorNote

User = get_user_model()

class HospitalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Hospital
        fields = ["id", "name"]

class DoctorLiteSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username"]

class PatientSerializer(serializers.ModelSerializer):
    hospital = HospitalSerializer(read_only=True)
    attending_doctor = DoctorLiteSerializer(read_only=True)

    class Meta:
        model = Patient
        fields = [
            "id", "first_name", "last_name", "date_of_birth", "gender",
            "hospital", "attending_doctor",
        ]

class EcgExamListSerializer(serializers.ModelSerializer):
    patient = PatientSerializer(read_only=True)

    class Meta:
        model = EcgExam
        fields = [
            "id", "status", "created_at", "updated_at",
            "patient", "top_predictions", "report_url", "preview_url",
        ]

class EcgExamDetailSerializer(EcgExamListSerializer):
    class Meta(EcgExamListSerializer.Meta):
        fields = EcgExamListSerializer.Meta.fields + ["model_version"]

# ✅ NEW: DoctorNote serializer
class DoctorNoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorNote
        fields = ["id", "patient", "text", "created_at", "updated_at", "author"]
        read_only_fields = ["id", "created_at", "updated_at", "author"]

# Current user payload for Flutter
class CurrentUserSerializer(serializers.ModelSerializer):
    hospital = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "username", "first_name", "last_name", "email", "is_staff", "is_superuser", "hospital", "role"]

    def get_hospital(self, obj):
        return None