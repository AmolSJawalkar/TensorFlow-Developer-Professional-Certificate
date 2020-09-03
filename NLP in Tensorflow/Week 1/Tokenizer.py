import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tf.config.set_visible_devices([], 'GPU')

sentences = [
    'Pranali loves Amol',
    'Amol loves Pranali'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)
print(tokenizer.index_word)