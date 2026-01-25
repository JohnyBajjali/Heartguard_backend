# HeartGuardBE/forms.py
from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import ReadOnlyPasswordHashField
from django.contrib.auth.password_validation import validate_password

User = get_user_model()

class CustomUserCreationForm(forms.ModelForm):
    """
    Used by the admin 'Add user' page.
    Shows password1/password2 and hashes on save().
    """
    password1 = forms.CharField(label="Password", widget=forms.PasswordInput)
    password2 = forms.CharField(label="Password confirmation", widget=forms.PasswordInput)

    class Meta:
        model = User
        # include the fields you want when creating a user in admin
        fields = ("username", "email", "first_name", "last_name", "role", "hospital", "is_staff", "is_active", "is_superuser")

    def clean_password1(self):
        pw = self.cleaned_data.get("password1")
        validate_password(pw)  # optional; remove if you don't want Django's validators
        return pw

    def clean(self):
        cleaned = super().clean()
        pw1 = cleaned.get("password1")
        pw2 = cleaned.get("password2")
        if pw1 and pw2 and pw1 != pw2:
            self.add_error("password2", "Passwords do not match.")
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])  # <-- hashes the password
        if commit:
            user.save()
            # if you have M2M like groups/permissions in 'fields', save_m2m after save:
            self.save_m2m()
        return user


class CustomUserChangeForm(forms.ModelForm):
    """
    Used by the admin 'Change user' page.
    Shows a read-only password hash and a link to the dedicated
    'Change password' form (admin/user/<id>/password/).
    """
    password = ReadOnlyPasswordHashField(
        label="Password",
        help_text=(
            "Raw passwords are not stored, so there is no way to see this user's password, "
            "but you can change the password using the dedicated form."
        ),
    )

    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name", "role", "hospital",
                  "is_staff", "is_active", "is_superuser", "groups", "user_permissions", "last_login", "date_joined")

    def clean_password(self):
        # Regardless of what the user provides, return the initial value.
        return self.initial.get("password")
