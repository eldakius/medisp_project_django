import os

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase
from machine_learning.models import Label, HistImage
from machine_learning.serializers import HistImageSerializer
from medisp_project.settings import HIST_IMAGES


class HistImageViewTestCase(TestCase):
    def setUp(self):
        Label.objects.all().delete()
        self.benign = Label.objects.create(name="benign")
        self.malignant = Label.objects.create(name="malignant")

        # self.create_sample_image(self.label_benign)
        def create_sample_image(self,label):
            sample_image= HistImage(label=label, file="medisp_storage/hist_images/train/benign/train_2.bmp")
            sample_image.save()
        def test_register_images_train_model(self):
            self.create_sample_image(self.benign)

            response_register= self.client.post(reverse("register-images"))
            self.assertEqual(response_register.status_code, status.HTTP_200_OK)

            response_train= self.client.post(reverse("train-model"))
            self.assertEqual(response_train.status_code, status.HTTP_200_OK)




# class TestHistImageModelViewset(TestCase):
#     @classmethod
#     def setUpClass(cls):
#         super().setUpClass()
#         cls.client = APIClient()
#
#         label1 = Label.objects.create(name="benign")
#         label2 = Label.objects.create(name="malignant")
#
#         cls.label1 = label1
#         cls.label2 = label2
#
#     def test_register_images(self):
#         url = reverse("histimagemodelviewset-register-images")
#         response = self.client.post(url)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#
#     def test_train_classification_model(self):
#         url = reverse("histimagemodelviewset-train-model")
#         response = self.client.post(url)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#
#
# class TestLabel(TestCase):
#     def setUp(self):
#         self.client = APIClient()
#
#         self.label1 = Label.objects.create(name="Label 1")
#         self.label2 = Label.objects.create(name="Label 2")
#
#     def test_get_label(self):
#         url = reverse("labelmodelviewset-detail", kwargs={"pk": self.label1.pk})
#         response = self.client.get(url)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#
#     def test_create_label(self):
#         url = reverse("labels-list")
#         data = {"name": "New Label"}
#         response = self.client.post(url, data=data)
#         self.assertEqual(response.status_code, status.HTTP_201_CREATED)


# class TestHistImageModelViewset(APITestCase):
#
#     def setUp(self) -> None:
#
#         self.user = User.objects.create_user(
#
#             username="test_user", password="test_user", email="test@user.com"
#
#         )
#
#         self.client.force_authenticate(user=self.user)
