from django.apps import AppConfig


class HeartguardbeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'HeartGuardBE'
    verbose_name = 'HeartGuard'   # 👈 add this line
