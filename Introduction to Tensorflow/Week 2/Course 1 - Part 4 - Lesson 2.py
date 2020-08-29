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


class TensorFlowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > 0.9:
            print("\nAccuracy reached 90%. Stopping the training.")
            self.model.stop_training = True


(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print("Training Images :" + str(training_images.shape))
print("Test Images :" + str(test_images.shape))

plt.imshow(training_images[0])

training_x = training_images/255
test_x = test_images/255

# Build Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

callback = TensorFlowCallback()
model.fit(x=training_x, y=training_labels, epochs=15, callbacks=[callback])

print(model.evaluate(x=test_x, y=test_labels))
