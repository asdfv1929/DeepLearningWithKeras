# 学习率指数衰减

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler
from math import pow, floor

dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

seed = 7
np.random.seed(seed)

def step_decay(epoch):
	init_lrate = 0.1
	drop = 0.5
	epochs_drop = 10
	lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
	return lrate

def create_model(init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
	
	learningRate = 0.1
	momentum = 0.9
	decay_rate = 0.0
	
	# 定义学习率衰减
	sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)
	
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model
	
# 学习率指数衰减回调
lrate = LearningRateScheduler(step_decay)
epochs = 200
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=5, verbose=1, callbacks=[lrate])
model.fit(x, Y)