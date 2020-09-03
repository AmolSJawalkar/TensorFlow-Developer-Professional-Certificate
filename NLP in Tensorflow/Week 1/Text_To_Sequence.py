import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
tf.config.set_visible_devices([], 'GPU')

sentences = [
    'How are you?',
    'I am good, you?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

sequence = tokenizer.texts_to_sequences(sentences)
print(sequence)
