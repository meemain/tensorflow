# Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad.
# Create a convolutional neural network that trains to 100% accuracy on these images, which cancels training
# upon hitting training accuracy of >.999
#
# Hint -- it will work best with 3 convolutional layers.
# curl https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip -O /tmp/happy-or-sad.zip

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
import zipfile

DESIRED_ACCURACY = 0.999

zip_ref = zipfile.ZipFile("tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("tmp/h-or-s")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if (logs.get('accuracy')>DESIRED_ACCURACY):
          print('Reached 99.9% accuracy so cancelling training!')
          self.model.stop_training=True

callbacks = myCallback()

# This Code Block should Define and Compile the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy']
              )

# This code block should create an instance of an ImageDataGenerator called train_datagen
# And a train_generator by calling train_datagen.flow_from_directory


train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'tmp/h-or-s/',
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)


# Expected output: 'Found 80 images belonging to 2 classes'

# This code block should call model.fit and train for
# a number of epochs.
history = model.fit(
    train_generator,
    steps_per_epoch=5,
    epochs=15,
    callbacks=[callbacks],
    verbose=1
)

model.save('happy_sad.h5')

