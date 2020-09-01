import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

inceptionModel = tf.keras.applications.inception_v3.InceptionV3(include_top=False, input_shape=(150, 150, 3), weights=None)
inceptionModel.load_weights('../data/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
for layer in inceptionModel.layers:
    layer.trainable = False

lastlayer = inceptionModel.get_layer('mixed10')

x = tf.keras.layers.Flatten()(lastlayer.output)
x = tf.keras.layers.Dense(units=512, activation='relu')(x)
x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

ourmodel = tf.keras.models.Model(inputs=inceptionModel.inputs, outputs=x)
ourmodel.compile(loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

train_DataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./ 255,
    rotation_range=40,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True
)

valid_DataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./ 255
)

sourcedir = '../data/cats_and_dogs_filtered'
trainImageDataGenerator = train_DataGenerator.flow_from_directory(
    sourcedir + '/train',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

validImageDataGenerator = train_DataGenerator.flow_from_directory(
    sourcedir + '/validation',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

history = ourmodel.fit(trainImageDataGenerator, epochs=10, validation_data=validImageDataGenerator)
