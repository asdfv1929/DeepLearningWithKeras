# 自动保存最优模型

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

Y_labels = to_categorical(Y, num_classes=3)

seed = 7
np.random.seed(seed)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
	
model = create_model()

# 设置检查点，仅保存为一个文件weight.best.h5
filepath = 'weight.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x, Y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0, callbacks=callback_list)
