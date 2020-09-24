import numpy as np
import os
import cv2
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D, Input, Dense, Flatten, Dropout
# from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from resnet import *

data = []
labels = []
imagePaths = list(paths.list_images("sport"))

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[0]

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# augmentation
trainAug = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                              zoom_range=0.15, fill_mode="nearest")
valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean
train_im = trainX
class_types = len(lb.classes_)

baseModel = resnet50()
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.4)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()

for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss="categorical_crossentropy", optimizer=SGD(), metrics="accuracy")

# train model
H = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=32), steps_per_epoch=len(trainX)//32,
                        validation_data=valAug.flow(testX, testY), validation_steps=len(testX)//32, epochs=20)

model.save("sport/model.h5")
