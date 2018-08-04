from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import pandas as pd

train = pd.read_csv('data/shuttle.trn', delim_whitespace=True, header=None)
test = pd.read_csv('data/shuttle.tst', delim_whitespace=True, header=None)

train_input = train[train.columns[range(9)]]
train_output = train[train.columns[9]]

test_input = test[test.columns[range(9)]]
test_output = test[test.columns[9]]

number_of_output_classes = pd.unique(train_output.values)

model = Sequential([
    Dense(32, input_shape=(len(train_input.columns),)),
    Activation('relu'),
    Dense(8),
    Activation('softmax'),
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(train_output)

model.fit(train_input, one_hot_labels, epochs=10, batch_size=32)
