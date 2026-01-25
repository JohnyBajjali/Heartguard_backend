from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import generics, permissions
from .serializers_user_create import RegisterSerializer

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, trim_whitespace=False)

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password')

    def validate_password(self, value):
        validate_password(value)     # optional: use Django’s validators
        return value

    def create(self, validated_data):
        password = validated_data.pop('password')
        # This call hashes the password:
        user = User.objects.create_user(password=password, **validated_data)
        return user

    class RegisterView(generics.CreateAPIView):
        permission_classes = [permissions.AllowAny]
        serializer_class = RegisterSerializer

    # If you ever allow updating password via this serializer:
    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for k, v in validated_data.items():
            setattr(instance, k, v)
        if password:
            instance.set_password(password)
        instance.save()
        return instance
