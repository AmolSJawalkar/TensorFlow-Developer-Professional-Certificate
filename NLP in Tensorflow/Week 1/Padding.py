import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.config.set_visible_devices([], 'GPU')

sentences = [
    'How are you?',
    'I am good, how about you?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

sequence = tokenizer.texts_to_sequences(sentences)

print("padded_sequence : "                  + str(pad_sequences(sequence)))
print("padded_sequence post : "             + str(pad_sequences(sequence, padding='post')))
print("padded_sequence Max length(Pre) : "  + str(pad_sequences(sequence, padding='post', maxlen=5)))
print("padded_sequence Truncate Post: "     + str(pad_sequences(sequence, padding='post', maxlen=5, truncating='post')))
