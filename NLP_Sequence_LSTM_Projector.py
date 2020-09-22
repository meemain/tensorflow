
import io
import json
import tensorflow as tf
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=160000 #Your dataset size here. Experiment using smaller values (i.e. 16000), but don't forget to train on at least 160000 to see the best effects
test_portion=.1

corpus = []
num_sentences = 0

with open("tmp/twitter_sentiments/training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        # Your Code here. Create list items where the first item is the text,
        # found in row[5], and the second is the label.
        # Note that the label is a '0' or a '4' in the text. When it's the former, make
        # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
        list_item=[]
        list_item.append(row[5])
        this_label = row[0]
        if this_label == '0':
            list_item.append(0)
        else:
            list_item.append(1)

        num_sentences = num_sentences + 1
        corpus.append(list_item)

sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size=len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]
model = tf.keras.models.load_model('tmp/twitter_sentiments/twitter_sentiments_sequence.h5')

#embedding projector preparation
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
out_v = io.open('tmp/twitter_sentiments/twitter_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('tmp/twitter_sentiments/twitter_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embedding = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embedding]) + "\n" )
out_v.close()
out_m.close()