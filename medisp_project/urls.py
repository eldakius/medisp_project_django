"""
URL configuration for medisp_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from machine_learning import views
from machine_learning.views import HistImageModelViewset

# from django.urls import path


urlpatterns = [
    path("admin/", admin.site.urls),
]


router = routers.DefaultRouter()
router.register(r"users", views.UserViewSet)
router.register(r"groups", views.GroupViewSet)
router.register(r"labels", views.LabelModelViewset, basename="labelmodelviewset")
router.register(r"hist-image", views.HistImageModelViewset, basename="histimage")


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include(router.urls)),
    path("api-auth/", include("rest_framework.urls", namespace="rest_framework")),
    path(
        "labels/",
        views.LabelModelViewset.as_view({"post": "create"}),
        name="labels-list",
    ),
    path(
        "hist-images/register-images/",
        HistImageModelViewset.as_view({"post": "register_images"}),
        name="histimagemodelviewset-register-images",
    ),
    path(
        "hist-images/train-model/",
        HistImageModelViewset.as_view({"post": "train_classification_model"}),
        name="histimagemodelviewset-train-model",
    ),
]
