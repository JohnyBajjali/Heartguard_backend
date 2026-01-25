# HeartGuardBE/tests.py
from django.urls import reverse
from rest_framework.test import APIClient, APITestCase
from django.contrib.auth import get_user_model
from .models import Patient

User = get_user_model()

class Smoke(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="doc", password="docpass")
        self.client = APIClient()
        # obtain JWT
        res = self.client.post(reverse("token_obtain_pair"), {"username": "doc", "password": "docpass"}, format="json")
        self.assertEqual(res.status_code, 200)
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {res.data['access']}")

    def test_me_and_patients(self):
        self.assertEqual(self.client.get("/api/users/me/").status_code, 200)
        Patient.objects.create(first_name="Ali", last_name="Saleh")
        res = self.client.get("/api/patients/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.data)
