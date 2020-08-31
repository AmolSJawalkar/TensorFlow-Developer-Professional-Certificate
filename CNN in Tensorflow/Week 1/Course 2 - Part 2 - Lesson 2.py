import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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
    sourcedir + '/train',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)
validDataGenerator = imageDataGenerator.flow_from_directory(
    sourcedir + '/validation',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

# Train the model
history = model.fit(trainDataGenerator,
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=validDataGenerator,
                    validation_steps=50)

print(history.history)

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





