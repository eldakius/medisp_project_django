from django.contrib.auth.models import User, Group
from rest_framework import permissions
from rest_framework.decorators import action
from medisp_project import settings
from .models import Label, HistImage
from .serializers import (
    UserSerializer,
    GroupSerializer,
    LabelSerializer,
    HistImageSerializer,
)
from rest_framework.response import Response
from rest_framework import status, viewsets
from PIL import Image as PILImage
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


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
        server_directory = "medisp_storage/hist_images"
        label_map = {
            "malignant": "malignant",
            "benign": "benign",
        }
        registered_data = []

        for label_folder, label_name in label_map.items():
            label, _ = Label.objects.get_or_create(name=label_name)
            label_path = os.path.join(server_directory, label_folder)

        for filename in os.path.join(server_directory):
            if filename.endswith(".bmp"):
                hist_image_path = os.path.join(label_folder, filename)
                # registered_data.append((hist_image_path, label.id))

                hist_image_dict = {
                    "hist_image": open(hist_image_path, "rb"),
                    "label": label.id,
                }

                hist_image_serialized = self.get_serializer(data=hist_image_dict)

                hist_image = HistImage(label=label, file=hist_image_path)
                hist_image.save()

        return Response({"status": "success"}, status=status.HTTP_200_OK)

    @action(
        methods=["post"],
        detail=False,
        url_path="train-model",
    )
    def train_classification_model(self, request):
        image_data = []
        labels = []

        hist_images = HistImage.objects.all()

        for hist_image in hist_images:
            labels.append(hist_image.label.id)
            train_image=cv2.imread(hist_image.file.path)

            if train_image is not None:
                train_image = cv2.resize(train_image, (128, 128))
                image_data.append(train_image)

            image_data = np.array(image_data)

            s = np.arange(image_data.shape[0])
            np.random.shuffle(s)
            tdata = image_data[s]
            labels = labels[s]

            data_length = len(image_data)

            (x_train, x_test) = (
                image_data[(int)(0.1 * data_length):],
                image_data[: (int)(0.1 * data_length)],
            )
            x_train = x_train.astype("float32") / 255
            x_test = x_test.astype("float32") / 255

            (y_train, y_test) = (
                labels[(int)(0.1 * data_length):],
                image_data[: (int)(0.1 * data_length)],
            )

            model = models.Sequential()
            model.add(
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3))
            )
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation="relu"))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation="relu"))

            model.summary()

            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.Dense(10))

            model.summary()

            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
            history = model.fit(
                x_train, y_train, epochs=50, validation_data=(x_test, y_test)
            )

            model.save(os.path.join(settings.BASE_DIR, "model.h5"))

            return Response({"status": "success"}, status=status.HTTP_200_OK)
