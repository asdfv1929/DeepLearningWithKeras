# 从检查点导入模型

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

Y_labels = to_categorical(Y, num_classes=3)

seed = 7
np.random.seed(seed)

def load_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
	
	filepath = 'weight.best.h5'
	model.load_weights(filepath=filepath)
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
	
model = load_model()

scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))