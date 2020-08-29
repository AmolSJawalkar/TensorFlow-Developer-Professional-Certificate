import tensorflow as tf

# Below code is to  Disable all GPUS as it take much time on my PC to detect it
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


# Call back to stop training
class TensorFlowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["accuracy"] > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


# Get the data
mnist = tf.keras.datasets.mnist
(trainig_images, training_labels), (test_images, test_labels) = mnist.load_data()

print("Training Images : " + str(trainig_images.shape))

# Normalize the input features
traning_x = trainig_images/255
test_x = test_images/255

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

callback = TensorFlowCallback()
model.fit(x=traning_x, y=training_labels, epochs=20, callbacks=[callback])
model.evaluate(x=test_x, y=test_labels)