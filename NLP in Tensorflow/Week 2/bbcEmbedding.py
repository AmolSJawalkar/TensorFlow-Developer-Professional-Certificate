import csv
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.config.set_visible_devices([], 'GPU')

categories = []
articles = []

with open('../data/bbc-text.csv', mode='r') as file:
    csvReader = csv.reader(file)
    next(csvReader)  # skipping the header

    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
    for row in csvReader:
        categories.append(row[0])
        article = row[1]
        for stopword in stopwords:
            word = ' ' + stopword + ' '
            article = article.replace(word, ' ')
            article = article.replace('  ', ' ')

        articles.append(article)


split = 0.8
vocabSize = 1000
maxLength = 120
embeddingSize = 16

sampleSize = len(articles)
partion = math.ceil(sampleSize * split)
texts_train = articles[: partion]
texts_test = articles[partion:]
labels_train = categories[: partion]
labels_test = categories[partion:]

tokenizer = Tokenizer(oov_token='<OOV>', num_words=vocabSize)
tokenizer.fit_on_texts(texts_train)

sequences_train = tokenizer.texts_to_sequences(texts_train)
padded_train = pad_sequences(sequences_train, maxlen=maxLength, padding='post', truncating='post')

sequences_test = tokenizer.texts_to_sequences(texts_test)
padded_test = pad_sequences(sequences_test, maxlen=maxLength, padding='post', truncating='post')

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(categories)

labels_train_seq = np.array(label_tokenizer.texts_to_sequences(labels_train))
labels_test_seq = np.array(label_tokenizer.texts_to_sequences(labels_test))

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocabSize, embeddingSize, input_length=maxLength))
model.add(tf.keras.layers.Flatten()),
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=(len(label_tokenizer.word_index) + 1), activation='softmax'))

model.summary()

# compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(padded_train, labels_train_seq, validation_data=(padded_test, labels_test_seq), epochs=10)

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