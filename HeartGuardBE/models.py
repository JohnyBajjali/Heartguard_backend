# HeartGuardBE/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.validators import RegexValidator

User = get_user_model()


class TimeStampedModel(models.Model):
    """Base model to add created/updated timestamps to all records."""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Hospital(TimeStampedModel):
    """A tenant-like unit (optional for now)."""
    name = models.CharField(max_length=200, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class UserProfile(TimeStampedModel):
    """Optional profile for each user to hold role + hospital scope."""

    class Roles(models.TextChoices):
        SUPERADMIN = "SUPERADMIN", "Super Admin"
        HOSPITAL_ADMIN = "HOSPITAL_ADMIN", "Hospital Admin"
        DOCTOR = "DOCTOR", "Doctor"
        STAFF = "STAFF", "Staff"

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=20, choices=Roles.choices, default=Roles.STAFF)
    hospital = models.ForeignKey(
        Hospital, null=True, blank=True, on_delete=models.SET_NULL, related_name="users"
    )

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self) -> str:
        return f"{self.user.username} ({self.role})"


@receiver(post_save, sender=User)
def create_profile_for_user(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


# Only letters and spaces (keep as-is)
patient_name_validator = RegexValidator(
    regex=r"^[A-Za-z\s]+$",
    message="Name must contain letters and spaces only.",
)


class Patient(TimeStampedModel):
    """
    Patient for mobile login + optional dashboard fields.

    IMPORTANT:
    - ECGUpload table does NOT store patient owner
    - Patient stores history as a list of ECGUpload IDs in ecg_upload_ids
    """

    patient_id = models.PositiveIntegerField(unique=True)
    full_name = models.CharField(
        max_length=255,
        validators=[patient_name_validator],
        help_text="Letters and spaces only.",
    )

    # History of ECGUpload IDs (newest first)
    ecg_upload_ids = models.JSONField(default=list, blank=True)

    class Gender(models.TextChoices):
        MALE = "M", "Male"
        FEMALE = "F", "Female"
        OTHER = "O", "Other"

    first_name = models.CharField(max_length=120, blank=True, default="")
    last_name = models.CharField(max_length=120, blank=True, default="")
    date_of_birth = models.DateField(null=True, blank=True)
    age = models.PositiveSmallIntegerField(null=True, blank=True)

    # gender optional for some flows
    gender = models.CharField(
        max_length=1,
        choices=Gender.choices,
        null=True,
        blank=True,
        default=None,
    )

    hospital = models.ForeignKey(
        Hospital, null=True, blank=True, on_delete=models.CASCADE, related_name="patients"
    )
    attending_doctor = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="patients_under_care",
    )

    mrn = models.CharField(max_length=64, blank=True, default="", help_text="MRN (optional)")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["hospital"]),
            models.Index(fields=["patient_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.patient_id} - {self.full_name}"


class EcgExam(TimeStampedModel):
    """Kept as-is (legacy/optional AI exam record)."""

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        PROCESSING = "PROCESSING", "Processing"
        READY = "READY", "Ready"
        FAILED = "FAILED", "Failed"

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="ecg_exams")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)

    top_predictions = models.JSONField(default=list, blank=True)

    device_vendor = models.CharField(max_length=80, blank=True, default="")
    device_model = models.CharField(max_length=80, blank=True, default="")
    source_filename = models.CharField(max_length=255, blank=True, default="")
    uploaded_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name="uploaded_ecgs"
    )

    report_url = models.URLField(null=True, blank=True)
    preview_url = models.URLField(null=True, blank=True)

    inference_started_at = models.DateTimeField(null=True, blank=True)
    inference_finished_at = models.DateTimeField(null=True, blank=True)
    model_version = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["patient", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"ECG Exam #{self.pk} - {self.patient} - {self.status}"


class ECGUpload(models.Model):
    """
    Global database of ECG files (NO patient owner stored here).
    Stores prediction results so patient history can show it.
    """

    class SourceType(models.TextChoices):
        XML = "xml", "XML"
        AECG = "aecg", "aECG"
        IMAGE_3X4 = "image_3x4", "3x4 Image"
        IMAGE_3X4P1 = "image_3x4+1", "3x4+1 Image"

    file = models.ImageField(upload_to="ecg_uploads/")
    source = models.CharField(max_length=50, default="mobile")

    # ✅ upload type (xml / aecg / image_3x4 / image_3x4+1)
    source_type = models.CharField(
        max_length=32,  # ✅ changed from 20 -> 32
        choices=SourceType.choices,
        null=True,      # ✅ recommended
        blank=True,     # ✅ recommended
        default=SourceType.IMAGE_3X4,
    )

    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Saved prediction output
    prediction_topk = models.JSONField(null=True, blank=True)  # list[{label,p}]
    prediction_meta = models.JSONField(null=True, blank=True)  # dict
    predicted_at = models.DateTimeField(null=True, blank=True)

    # Preview image url (xml preview, digitized preview, ...)
    preview_image_url = models.URLField(null=True, blank=True)

    def __str__(self):
        return f"ECGUpload #{self.pk} - {self.file.name} ({self.uploaded_at})"


class DoctorNote(TimeStampedModel):
    """Multiple notes per patient, editable, with dates."""

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="doctor_notes")
    text = models.TextField()

    author = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name="doctor_notes"
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["patient", "created_at"]),
        ]

    def __str__(self):
        return f"DoctorNote #{self.pk} for Patient {self.patient.patient_id}"