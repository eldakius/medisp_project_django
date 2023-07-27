from django.contrib.auth.models import User, Group
from rest_framework import serializers, viewsets

from machine_learning.models import Label, HistImage


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ["url", "username", "email", "groups"]


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ["url", "name"]


class LabelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Label
        fields = "__all__"


class HistImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = HistImage
        fields = "__all__"


class LabelModelViewset(viewsets.ModelViewSet):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
