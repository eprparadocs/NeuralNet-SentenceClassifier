from __future__ import print_function

import os
import sys

import numpy as np
import keras

from sentence_types import load_encoded_data
from sentence_types import encode_data
from sentence_types import get_custom_test_comments

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

# Where we store the model data
model_name      = "models/cnn"

# Where we get training/test data
embedding_name  = "data/default"

# Model configuration
max_words = 110000
maxlen = 500
batch_size = 64
embedding_dims = 75
filters = 100
kernel_size = 5
hidden_dims = 350
epochs = 7

# Add parts-of-speech to data
pos_tags_flag = True


# Export & load embeddings
x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8, embedding_name=embedding_name, pos_tags=pos_tags_flag)
num_classes = np.max(y_train) + 1
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Construct model
model = Sequential()
model.add(Embedding(max_words, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Generate the network
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Write the model to the file system.
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_name + ".h5")

# Figure out how good a model/network we have...use the test data from the
# original input data
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Model Accuracy:', score[1])

test_comments, test_comments_category = get_custom_test_comments()

x_test, _, y_test, _ = encode_data(test_comments, test_comments_category, data_split=1.0, 
                                   embedding_name=embedding_name, add_pos_tags_flag=pos_tags_flag)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_test = keras.utils.to_categorical(y_test, num_classes)
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Manual test')
print('Test accuracy:', score[1])

# Show predictions against some of the test data
print(len(x_test))
predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
real = []
test = []
for i in range(0, len(predictions)):
    real.append(y_test[i].argmax(axis=0))
    test.append(predictions[i].argmax(axis=0))
print("Predictions")
print("Real", real)
print("Test", test)
