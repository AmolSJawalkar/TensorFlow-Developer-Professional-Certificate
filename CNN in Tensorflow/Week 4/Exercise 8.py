import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')


def get_data(csv_filePath):
    labels = []
    data = []
    with open(csv_filePath, mode='r') as csvFile:
        csv_reader = csv.reader(csvFile)
        lineCount = 0
        for row in csv_reader:
            if lineCount != 0:
                labels.append(int(row[0]))
                data.append(np.array(row[1:], dtype=int).reshape(28, 28))
            lineCount = lineCount + 1

    return np.array(labels), np.array(data)


trainlabels, traindata = get_data('../data/sign_mnist/sign_mnist_train.csv')
testlabels, testdata = get_data('../data/sign_mnist/sign_mnist_test.csv')

traindata = np.expand_dims(traindata, axis=-1)
testdata = np.expand_dims(testdata, axis=-1)

print(traindata.shape)
print(testdata.shape)

datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

trainImageDataGenerator = datagenerator.flow(x=traindata, y=trainlabels, batch_size=40)
testImageDataGenerator = datagenerator.flow(x=testdata, y=testlabels, batch_size=20)

# Build a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=25, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )


class TensorflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] >= 0.99:
            print('Accruacy reached 99%. Stopping the training')
            self.model.stop_training = True


callback = TensorflowCallback()
# Train the model
model.fit(x=trainImageDataGenerator,
          validation_data=testImageDataGenerator,
          epochs=20,
          callbacks=callback
          )

