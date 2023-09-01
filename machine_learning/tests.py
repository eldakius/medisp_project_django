from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from machine_learning.models import Label


class TestHistImageModelViewset(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = APIClient()

        label1 = Label.objects.create(name="benign")
        label2 = Label.objects.create(name="malignant")

        cls.label1 = label1
        cls.label2 = label2

    def test_register_images(self):
        url = reverse("histimagemodelviewset-register-images")
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_train_classification_model(self):
        url = reverse("histimagemodelviewset-train-model")
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class TestLabel(TestCase):
    def setUp(self):
        self.client = APIClient()

        self.label1 = Label.objects.create(name="Label 1")
        self.label2 = Label.objects.create(name="Label 2")

    def test_get_label(self):
        url = reverse("labelmodelviewset-detail", kwargs={"pk": self.label1.pk})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_create_label(self):
        url = reverse("labels-list")
        data = {"name": "New Label"}
        response = self.client.post(url, data=data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
