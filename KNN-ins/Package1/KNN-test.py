from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
def image_to_feature_vector(image, size=(50, 50)):
	# đưa ảnh thô về size nhỏ
	return cv2.resize(image, size).flatten()
def extract_color_histogram(image, bins=(8, 8, 8)):
	# trích xuất biểu đồ màu,
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)
	# trả về vector phẳng
	return hist.flatten()
args={
    "dataset":"data/train",
    "neighbors":1,
    "jobs":-1
}
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
# tạo các ma trận pixel và list nhãn.
rawImages = []
features = []
labels = []
# Lặp qua các ảnh vào
for (i, imagePath) in enumerate(imagePaths):
	#load từng ảnh và gán nhãn
	image = cv2.imread(imagePath)
	if(int(imagePath.split(os.path.sep)[-1].split(".")[0])<= 247):
		label="Defective"
	else:
		label="Normal"
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# cập nhật đữ liệu vào các list
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# update mỗi lần load đc 100 ảnh
	if i > 0 and i % 100 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
    rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
    features.nbytes / (1024 * 1000.0)))
# chia bộ dữ liệu thành 2 phần .
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.75, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.75, random_state=42)
print("[INFO] evaluating raw pixel accuracy...")
#KNN với kiểu ảnh thô
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
print("[INFO] evaluating histogram accuracy...")
#knn với kiểu ảnh
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))