# HeartGuardBE/serializers.py
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Hospital, Patient, EcgExam

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

# Current user payload for Flutter
class CurrentUserSerializer(serializers.ModelSerializer):
    # Keep shape tolerant; hospital is optional/None unless you model it on user
    hospital = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "username", "first_name", "last_name", "email", "is_staff", "is_superuser", "hospital", "role"]
        # 'role' only if your custom user has it; serializer will ignore if not present.

    def get_hospital(self, obj):
        # If you later relate user->hospital, return it here
        return None
