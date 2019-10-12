import os
import zipfile
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2
import argparse
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from tensorflow.python.lib.io import file_io


def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


def main(job_dir):
    EPOCHS = 100
    INIT_LR = 1e-3
    BS = 50
    IMAGE_DIMS = (96, 96, 3)

    data = []
    labels = []
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print('downloading!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    with file_io.FileIO('gs://data_bbp/data.zip', mode='wb+') as f:
        with file_io.FileIO(current_dir + '/data.zip', mode='wb+') as output_f:
            output_f.write(f.read())
    print('downloaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    with zipfile.ZipFile(current_dir + '/data.zip', 'r') as f:

        for member in f.infolist():
            f.extract(member, current_dir + '/data')

    print('unzipped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    imagePaths = sorted(list(paths.list_images(current_dir)))

    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    print('processed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.1, random_state=42)

    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=False, fill_mode="nearest")

    model = build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                  depth=IMAGE_DIMS[2], classes=len(lb.classes_))
    model.summary()

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='rb') as input_f:
        with file_io.FileIO('gs://bbp_model_bucket/model/model.h5', mode='wb+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)

# gcloud ai-platform jobs submit training a33 \
# --module-name=trainer.train \
# --package-path=./trainer/ \
# --job-dir=gs://bbp_model_bucket \
# --region=europe-west1 \
# --config=config.yaml \
# --python-version 3.5 \
# --runtime-version 1.13

# gcloud ml-engine local train --module-name trainer.train \
#           --package-path ./trainer/ \
#           --job-dir=gs://bbp_model_bucket
