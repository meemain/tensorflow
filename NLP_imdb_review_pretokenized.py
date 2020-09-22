import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import io
import matplotlib.pyplot as plt

imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder
print(tokenizer.subwords)
sample_string = "What a wonderful world this could be!"
tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print('Orignial string is {}'.format(original_string))

for ts in tokenized_string:
    print('{} --> {}'.format(ts,tokenizer.decode([ts])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

embedding_dims=64
num_epochs=10

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dims),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)



def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


#export vectors and metadata
e=model.layers[0]
weights=e.get_weights()[0]
print(weights.shape)

v_out = io.open('tmp/imdb_vecs.tsv', 'w', encoding='utf-8')
m_out = io.open('tmp/imdb_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, tokenizer.vocab_size):
    word=tokenizer.decode([word_num])
    embeddings=weights[word_num]
    m_out.write(word + "\n")
    v_out.write('\t'.join([str(x) for x in embeddings])+'\n')
v_out.close()
m_out.close()

# load vector and meta file in https://projector.tensorflow.org/ and check spherize data

