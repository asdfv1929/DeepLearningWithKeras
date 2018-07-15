# 多分类实例：鸢尾花分类

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# 导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

# 定义随机种子
seed = 7
np.random.seed(seed)

def create_model(optimizer='adam', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim = 4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
	
	# 编译模型
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
	
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))
