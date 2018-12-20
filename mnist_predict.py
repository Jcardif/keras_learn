import keras
import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist


num_classes = 10
# the data, split between train and test sets
(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)



with tf.device('/cpu:0'):
    model = load_model('my_model.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
