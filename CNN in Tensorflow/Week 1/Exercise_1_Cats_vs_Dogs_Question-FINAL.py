import os
import shutil
import math
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def splitdata(sourcedir, traindir, testdir, splitratio):
    def copydata(source, files, target):
        if not os.path.exists(target):
            os.makedirs(target)

        for filename in files:
            if os.path.getsize(source + '/' + filename) != 0:
                shutil.copyfile(source + '/' + filename, target + '/' + filename)
            else:
                print(filename + " is zero length. Ignoring it.\n")

    files = os.listdir(sourcedir)
    trainsize = math.ceil(len(files) * splitratio)
    copydata(sourcedir, files[:trainsize], traindir)
    copydata(sourcedir, files[trainsize:], testdir)


print("Splitting the data into train and validation")
sourcedir = '../data/PetImages/Cat'
traindir = '../data/cats_and_dogs/train'
testdir = '../data/cats_and_dogs/valid'
splitdata(sourcedir, traindir + '/cats', testdir + '/cats', 0.9)
splitdata(sourcedir, traindir + '/dogs', testdir + '/dogs', 0.9)
print("Split complete")

# Below code is to  Disable all GPUS as it take much time on my PC to detect it
tf.config.set_visible_devices([], 'GPU')

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

# Preprocess the input images
imageDataGenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
trainDataGenerator = imageDataGenerator.flow_from_directory(
    traindir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=50
)
validDataGenerator = imageDataGenerator.flow_from_directory(
    testdir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=50
)

# Train the model
history = model.fit(trainDataGenerator,
                    epochs=5,
                    validation_data=validDataGenerator)

print(history.history)

# loss, val_loss, accuracy, val_accuracy

# Let's plot the accuracy

epochs = range(len(history.history['accuracy']))
plt.plot(epochs, history.history['accuracy'])
plt.plot(epochs, history.history['val_accuracy'])
plt.title("Training and validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Let's plot the loss
plt.figure()
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title("Training and validation loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
