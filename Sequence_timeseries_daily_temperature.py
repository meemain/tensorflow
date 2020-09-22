#curl -o tmp/daily-min-temperatures.csv 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

import tensorflow as tf
from tensorflow import keras

import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    #plt.show()

time_step=[]
temps = []

with open('tmp/daily-min-temperatures.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    step = 0
    for row in reader:
        temps.append(float(row[1]))
        time_step.append(step)
        step = step+1

series = np.array(temps)
time = np.array(time_step)
plt.figure(figsize=(10,6))
plot_series(time, series)


split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size+1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast




#build and fit model with 100 epochs to get the optimum lr for optimizer
# tf.keras.backend.clear_session()
# tf.random.set_seed(51)
# np.random.seed(51)
# window_size = 64
# batch_size = 256
# train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# print(train_set)
# print(x_train.shape)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=5,
#                            strides=1, padding='causal',
#                            activation='relu',
#                            input_shape=[None, 1]),
#     tf.keras.layers.LSTM(64, return_sequences=True),
#     tf.keras.layers.LSTM(64, return_sequences=True),
#     tf.keras.layers.Dense(30, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1),
#     tf.keras.layers.Lambda(lambda x: x * 400)
# ])
#
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20)
# )
#
# optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
# model.compile(loss=tf.keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics="mae")
# history = model.fit(train_set, epochs=100, callbacks=[lr_schedule], verbose=1)

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([1e-8, 1e-4, 0, 60])
# plt.show()


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                           strides=1, padding='causal',
                           activation='relu',
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])


optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics="mae")
history = model.fit(train_set, epochs=150, verbose=1)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size: -1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()

print('MAE:', tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())

print('RNN_FORECAST:', rnn_forecast)



