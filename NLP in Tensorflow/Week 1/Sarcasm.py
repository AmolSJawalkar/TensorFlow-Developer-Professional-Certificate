import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

labels = []
headlines = []

with open('../data/sarcasm.json', mode='r') as file:
    data = json.load(file)
    for obj in data:
        labels.append(obj['is_sarcastic'])
        headlines.append(obj['headline'])


tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(headlines)
print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(headlines)
padded_sequence = pad_sequences(sequences, padding='post')
index = 2
print("Original : " + headlines[index])
print("Padded Sequence : " + str(padded_sequence[index]))
texts = tokenizer.sequences_to_texts(padded_sequence)
print("Text : " + str(texts[index]))
