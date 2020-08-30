import tensorflow as tf
import matplotlib.pyplot as plt

# Below code is to  Disable all GPUS as it take much time on my PC to detect it
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])


# Callbacks
class TensorflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] >= 0.999:
            print("Reached accuracy 99.9%, stopping the training")
            self.model.stop_training = True


# Preprocess the input images
imageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
trainDataGenerator = imageDataGenerator.flow_from_directory(
    'data/happy-or-sad',
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

callback = TensorflowCallback()
# Train the model
model.fit(
    x=trainDataGenerator,
    epochs=15,
    steps_per_epoch=5,
    callbacks=[callback]
)
