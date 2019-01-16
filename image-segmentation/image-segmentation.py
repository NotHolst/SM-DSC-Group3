import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Layer, Permute, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Reshape
import numpy as np


batch_size = 30
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.load('./output/trainingSet.npy')
y_train = np.load('./output/trainingSetLabels.npy')
x_test = np.load('./output/testSet.npy')
y_test = np.load('./output/testSetLabels.npy')

# x_train = x_train.reshape(37, 227, 27, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()

# Encoding Start
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation="relu", input_shape=(226, 226, 3), padding="valid"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())

# Decoding Start
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(2, 2)))
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"))
model.add(BatchNormalization())

model.add(Conv2D(3, kernel_size=(1, 1), padding="valid"))

# model.add(Reshape((3, model.output_shape[-2]*model.output_shape[-1]),
#                   input_shape=(3, model.output_shape[-2]*model.output_shape[-1])))
# model.add(Permute((2, 1)))

model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('./savedModels/test.h5', True)
