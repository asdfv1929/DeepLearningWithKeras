# 模型增量更新

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)

dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

x_train, x_increment, Y_train, Y_increment = train_test_split(x, Y, test_size=0.2, random_state=seed)

Y_train_labels = to_categorical(Y_train, num_classes=3)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
	
model = create_model()
model.fit(x_train, Y_train_labels, epochs=10, batch_size=5, verbose=2)

scores = model.evaluate(x_train, Y_train_labels, verbose=0)
print('Base %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

model_json = model.to_json()
with open('model.increment.json', 'w') as file:
	file.write(model_json)
	
model.save_weights('model.increment.json.h5')

with open('model.increment.json', 'r') as file:
	model_json = file.read()
	
new_model = model_from_json(model_json)
new_model.load_weights('model.increment.json.h5')

new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 增量训练模型
Y_increment_labels = to_categorical(Y_increment, num_classes=3)
new_model.fit(x_increment, Y_increment_labels, epochs=10, batch_size=5, verbose=2)
scores = new_model.evaluate(x_increment,  Y_increment_labels, verbose=0)
print('Increment %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))