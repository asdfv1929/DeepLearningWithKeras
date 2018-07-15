# json序列化模型

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json

dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# 将标签转换成分类编码
Y_labels = to_categorical(Y, num_classes=3)

seed = 7
np.random.seed(seed)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='sigmoid', kernel_initializer=init))
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
	
model = create_model()
model.fit(x, Y_labels, epochs=200, batch_size=5, verbose=0)

scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

# 将模型保存为JSON文件
model_json = model.to_json()
with open('model.json', 'w') as file:
	file.write(model_json)
	
# 保存模型的权重值
model.save_weights('model.json.h5')

with open('model.json', 'r') as file:
	model_json = file.read()
	
# 加载模型
new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')

new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

scores = new_model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))