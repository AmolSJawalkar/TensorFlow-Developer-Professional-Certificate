import tensorflow as tf
import numpy as np

# Below code is to  Disable all GPUS as it take much time on my PC to detect it
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=1, input_shape=tf.TensorShape([1])))
model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.fit(x=xs, y=ys, epochs=500)

print(model.predict([10, 12, 13]))
