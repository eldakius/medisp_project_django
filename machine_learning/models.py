from django.db import models


# Create your models here.
class Label(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(unique=True, max_length=120)


class Image(models.Model):
    id = models.AutoField(primary_key=True)
    label = models.ForeignKey(
        Label, on_delete=models.SET_NULL, related_name="rois", null=True
    )
    file = models.FileField(upload_to="hist_images", max_length=255)

    class Meta:
        abstract = True


class HistImage(Image):
    MAG40 = 40
    MAG100 = 100
    MAG200 = 200
    MAG400 = 400
    UNKNOWN = 0
    MAGS = (
        (MAG40, "40X"),
        (MAG100, "100X"),
        (MAG200, "200X"),
        (MAG400, "400X"),
        (UNKNOWN, "UNKNOWN"),
    )

    class Stain(models.TextChoices):
        HEMATOXYLINEOSIN = "H&E", "Hematoxylin eosin (H&E)"
        UNKNOWN = "", "Unknown"

    magnification = models.IntegerField(choices=MAGS, default=UNKNOWN)
    stain = models.CharField(max_length=50, choices=Stain.choices, blank=True)
