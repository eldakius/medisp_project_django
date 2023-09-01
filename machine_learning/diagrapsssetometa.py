# @action(
#     methods=["post"],
#     detail=False,
#     url_path="register-images",
# )
# def register_images(self, request):
#     tdata = []
#     labels = []
#     # na allaksw to onoma tis metavlitis
#     benign = os.listdir(
#         os.path.join(settings.BASE_DIR, "medisp_storage/hist_images/train/benign")
#     )
#     for x in benign:
#         img = cv2.imread(
#             os.path.join(
#                 settings.BASE_DIR, "medisp_storage/hist_images/train/benign", x
#             )
#         )
#         if img is not None:
#             img_from_ar = PILImage.fromarray(img, "RGB")
#             resized_image = img_from_ar.resize((50, 50))
#             tdata.append(np.array(resized_image))
#             labels.append(0)
#
#     # Malignant (label 1)
#     malignant = os.listdir(
#         os.path.join(
#             settings.BASE_DIR, "medisp_storage/hist_images/train/malignant"
#         )
#     )
#     for x in malignant:
#         img = cv2.imread(
#             os.path.join(
#                 settings.BASE_DIR, "medisp_storage/hist_images/train/malignant", x
#             )
#         )
#         if img is not None:
#             img_from_ar = PILImage.fromarray(img, "RGB")
#             resized_image = img_from_ar.resize((50, 50))
#             tdata.append(np.array(resized_image))
#             labels.append(1)
#
#     tdata_path = os.path.join(settings.BASE_DIR, "tdata.npy")
#     labels_path = os.path.join(settings.BASE_DIR, "labels.npy")
#     np.save(tdata_path, tdata)
#     np.save(labels_path, labels)
#
#     # tha kanei mia eggrafi gia kathe eikona me to label tis
#     HistImage()
#
#     return Response({"status": "success"}, status=status.HTTP_200_OK)

# for hist_image_path, label_id in registered_data:
#
#     tdata=os.path.basename(hist_image_path)
#     labels=os.path.basename(label_id)
