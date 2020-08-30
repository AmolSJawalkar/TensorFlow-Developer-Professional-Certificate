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

# Get the data
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# resize
training_x = training_images.reshape(60000, 28, 28, 1)
test_x = test_images.reshape(10000, 28, 28, 1)

# normalize the images
training_x = training_x / 255
test_x = test_x / 255

# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Add call back function to stop training
class TensorflowCallack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] >= 0.998:
            print("\n Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


callback = TensorflowCallack()
# train
model.fit(x=training_x, y=training_labels, epochs=20, callbacks=[callback])

# Evaluate
model.evaluate(x=test_x, y=test_labels)
