import shutil

from django.contrib.auth.models import User, Group
from rest_framework import viewsets, status
from rest_framework import permissions
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Label, HistImage
from .serializers import (
    UserSerializer,
    GroupSerializer,
    LabelSerializer,
    HistImageSerializer,
)

import tarfile
import os
import numpy as np
import tensorflow as tf
from PIL import Image


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = User.objects.all().order_by("-date_joined")
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """

    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


class LabelModelViewset(viewsets.ModelViewSet):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer


# Create your views here.
class HistImageModelViewset(viewsets.ModelViewSet):
    queryset = HistImage.objects.all()
    serializer_class = HistImageSerializer

    @action(
        methods=["post"],
        detail=False,
        url_path="register-images",
    )
    def register_images(self, request):
        hist_image_serialized = self.get_serializer()
        return Response(hist_image_serialized.data, status=status.HTTP_200_OK) @ action(
            detail=False, methods=["post"]
        )

        def train_classification_model(self, request):
            with tarfile.open("train.tar.xz", "r:xz") as tar:
                tar.extractall("temp")
                train_data = []
                train_labels = []
                train_data = np.array(train_data) / 255.0
                train_labels = np.array(train_labels)
            for dirpath, dirnames, filenames in os.walk("temp/train"):
                for filename in filenames:
                    if filename.endswith(".png"):
                        label = int(os.path.basename(dirpath))
                        img_path = os.path.join(dirpath, filename)
                        img = np.array(Image.open(img_path))
                        train_data.append(img)
                        train_labels.append(label)
            train_data = np.array(train_data) / 255.0
            train_labels = np.array(train_labels)

            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        32, (3, 3), activation="relu", input_shape=train_data.shape[1:]
                    ),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(10),
                ]
            )
            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
            model.fit(train_data, train_labels, epochs=10)

            shutil.rmtree("temp")

            return Response({"status": "success"})
