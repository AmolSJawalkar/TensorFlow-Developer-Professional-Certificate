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

# Build the model first
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # Means Zeor or One

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

# Let's preprocess the input images
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
trainGenerator = dataGenerator.flow_from_directory(
    'data/horse-or-human',
    target_size=(300, 300),  # This should match with input of the NN
    batch_size=64,
    class_mode='binary') # It means folder will have only two category

validGenerator = dataGenerator.flow_from_directory(
    'data/validation-horse-or-human',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)

model.fit_generator(trainGenerator,
                    steps_per_epoch=8,
                    epochs=10,
                    validation_data=validGenerator,
                    validation_steps=8)
