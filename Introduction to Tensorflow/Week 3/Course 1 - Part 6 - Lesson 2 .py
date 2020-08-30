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

# Get the images
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Reshape the input images to size (28x28x1)
training_x = training_images.reshape(60000, 28, 28, 1)
test_x = test_images.reshape(10000, 28, 28, 1)

# normalize it
training_x = training_x / 255.0
test_x = test_x / 255.0

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# Training the model
model.fit(x=training_x, y=training_labels, epochs=5)

# Evaluate model
model.evaluate(x=test_x, y=test_labels)

# Visualizing convolution and pooling
import matplotlib.pyplot as plt

layers = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layers)

CONV_FILTER_INDEX = [15, 15, 15, 15] # Filter 15th in eachh layer filter
image_Indexs = [4, 7, 26]  # Shirts

f, axarr = plt.subplots(len(image_Indexs), 4)  # no of images, no of layers
for row in range(len(image_Indexs)):
    image_index = image_Indexs[row]
    outPuts = activation_model.predict(test_images[image_index].reshape(1, 28, 28, 1))
    for layerIndex in range(0, 4):  # for first 4 layers
        image = outPuts[layerIndex]
        axarr[row, layerIndex].imshow(image[0, :, :, CONV_FILTER_INDEX[layerIndex]], cmap='inferno')
        axarr[row, layerIndex].grid(False)


print("Done")
