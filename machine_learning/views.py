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
        tdata = []
        labels = []
        # na allaksw to onoma tis metavlitis
        benign = os.listdir(
            os.path.join(settings.BASE_DIR, "medisp_storage/hist_images/train/benign")
        )
        for x in benign:
            img = cv2.imread(
                os.path.join(
                    settings.BASE_DIR, "medisp_storage/hist_images/train/benign", x
                )
            )
            if img is not None:
                img_from_ar = PILImage.fromarray(img, "RGB")
                resized_image = img_from_ar.resize((50, 50))
                tdata.append(np.array(resized_image))
                labels.append(0)

        # Malignant (label 1)
        malignant = os.listdir(
            os.path.join(
                settings.BASE_DIR, "medisp_storage/hist_images/train/malignant"
            )
        )
        for x in malignant:
            img = cv2.imread(
                os.path.join(
                    settings.BASE_DIR, "medisp_storage/hist_images/train/malignant", x
                )
            )
            if img is not None:
                img_from_ar = PILImage.fromarray(img, "RGB")
                resized_image = img_from_ar.resize((50, 50))
                tdata.append(np.array(resized_image))
                labels.append(1)

        tdata_path = os.path.join(settings.BASE_DIR, "tdata.npy")
        labels_path = os.path.join(settings.BASE_DIR, "labels.npy")
        np.save(tdata_path, tdata)
        np.save(labels_path, labels)

        # tha kanei mia eggrafi gia kathe eikona me to label tis
        HistImage()

        return Response({"status": "success"}, status=status.HTTP_200_OK)

    @action(
        methods=["post"],
        detail=False,
        url_path="train-model",
    )
    def train_classification_model(self, request):
        tdata_path = os.path.join(settings.BASE_DIR, "tdata.npy")
        labels_path = os.path.join(settings.BASE_DIR, "labels.npy")
        # diavazei ta hist_image apo to database
        # pws na ta xwrisei me to filter
        #
        tdata = np.load(tdata_path)
        labels = np.load(labels_path)

        s = np.arange(tdata.shape[0])
        np.random.shuffle(s)
        tdata = tdata[s]
        labels = labels[s]

        data_length = len(tdata)

        (x_train, x_test) = (
            tdata[(int)(0.1 * data_length) :],
            tdata[: (int)(0.1 * data_length)],
        )
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        (y_train, y_test) = (
            labels[(int)(0.1 * data_length) :],
            tdata[: (int)(0.1 * data_length)],
        )

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3)))
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
