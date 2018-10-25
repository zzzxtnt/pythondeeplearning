from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers


def vectorize_sequences(sequences, dimensiion=10000):
    results = np.zeros((len(sequences), dimensiion))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# print(train_data[0])
# print(train_label[0])
# print(x_train[0])

y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
print(history)
