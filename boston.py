from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_target)

mean = train_data.mean(axis=0)

train_data -= mean

std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_sample = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('process fold #', i)
    val_data = train_data[i * num_val_sample: (i + 1) * num_val_sample]
    val_target = train_target[i * num_val_sample:(i + 1) * num_val_sample]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_sample],
         train_data[(i + 1) * num_val_sample:]],
        axis=0)
    partial_train_target = np.concatenate(
        [train_target[:i * num_val_sample],
         train_target[(i + 1) * num_val_sample:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
    all_scores.append(val_mae)
    print(all_scores)

K.clear_session()

num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold ##', i)
    val_data = train_data[i * num_val_sample:(i + 1) * num_val_sample]
    val_target = train_target[i * num_val_sample:(i + 1) * num_val_sample]

    partial_train_data = np.concatenate([train_data[:i * num_val_sample], train_data[(i + 1) * num_val_sample:]],
                                        axis=0)

    partial_train_target = np.concatenate(
        [train_target[:i * num_val_sample], train_target[(i + 1) * num_val_sample:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_target, validation_data=(val_data, val_target),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

    print(all_mae_histories)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation Mae')
plt.show()
