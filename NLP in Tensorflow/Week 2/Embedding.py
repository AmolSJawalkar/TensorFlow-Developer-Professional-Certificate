import tensorflow as tf
import tensorflow_datasets as tds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

tf.config.set_visible_devices([], 'GPU')

# Load the dataset
imdbdataset = tds.load('imdb_reviews', as_supervised=True, )
traindataset = imdbdataset['train']
testdataset = imdbdataset['test']

texts_train = []
labels_train = []
for item in traindataset:
    texts_train.append(item[0].numpy().decode('utf8'))
    labels_train.append(item[1].numpy())

texts_test = []
labels_test = []
for item in testdataset:
    texts_test.append(item[0].numpy().decode('utf8'))
    labels_test.append(item[1].numpy())

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

# set the parameter
maxLength = 120
vocabSize = 25000
embedSize = 16

# Let's tokenize it
tokenizer = Tokenizer(num_words=vocabSize, oov_token='<OOV>')
tokenizer.fit_on_texts(texts_train)
print(tokenizer.word_index)
trainSequences = tokenizer.texts_to_sequences(texts_train)
padded_train = pad_sequences(trainSequences, maxlen=maxLength, padding='post', truncating='post')

testSequences = tokenizer.texts_to_sequences(texts_test)
padded_test = pad_sequences(testSequences, maxlen=maxLength, padding='post', truncating='post')


# Let's build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocabSize, embedSize, input_length=maxLength))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=6, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# Train the model
model.fit(x=padded_train,
          y=labels_train,
          epochs=15,
          validation_data=(padded_test, labels_test))

vecsfile = open("vecs.tsv", "w", encoding='utf-8')
metafile = open("meta.tsv", "w", encoding='utf-8')

embeddingLayer = model.layers[0]
weights = embeddingLayer.get_weights()[0]
print(weights.shape)
for index in range(1, vocabSize):
    word = tokenizer.index_word[index]
    metafile.write(word + '\n')
    embeddings = weights[index]
    vecsfile.write('\t'.join([str(x) for x in embeddings]) + "\n")

vecsfile.close()
metafile.close()







