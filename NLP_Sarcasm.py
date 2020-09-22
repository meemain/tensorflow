# curl -o tmp/sarcasm.json 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'

import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import numpy as np
import io

sarcasm=[]
#convert data to comma separated list for easy manipulation
#this dataset have 27000 records, lets use 20,000 for trainign and 7000 for validation
with open('tmp/NLP_SarcasmHeadline/Sarcasm_Headlines_Dataset_v2.json', 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        # converting string to json
        line_dict = eval(line)
        sarcasm.append(line_dict)

#datastore = json.load(sarcasm)
print(sarcasm[5])
sentences=[]
labels=[]
urls=[]

for items in sarcasm:
    sentences.append(items['headline'])
    labels.append(items['is_sarcastic'])
    urls.append(items['article_link'])

#hyper paramets
vocab_size=1000
embedding_dim=32
max_length=120
trunc_type= 'post'
padding_type='post'
oov_tok='<OOV>'
training_size=20000
num_epochs=30

training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]
training_labels=labels[0:training_size]
testing_labels=labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded=np.array(training_padded)
training_labels=np.array(training_labels)
testing_padded=np.array(testing_padded)
testing_labels=np.array(testing_labels)

#helper function
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(), #You could use Flatten() here as well
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(training_padded, training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


#generate vectors and meta data for Embedding Projector
e=model.layers[0]
weights=e.get_weights()[0]
print(weights.shape)

#export vectors and metadata
v_out = io.open('tmp/sarcasm_vecs.tsv', 'w', encoding='utf-8')
m_out = io.open('tmp/sarcasm_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word=reverse_word_index[word_num]
    embeddings=weights[word_num]
    m_out.write(word + "\n")
    v_out.write('\t'.join([str(x) for x in embeddings])+'\n')
v_out.close()
m_out.close()

