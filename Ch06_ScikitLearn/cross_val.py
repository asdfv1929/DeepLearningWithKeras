# Section 6.1 使用交叉验证评估模型

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

# 构建模型
def create_model():
	model = Sequential();
	model.add(Dense(units=12, input_dim=8, activation='relu'))
	model.add(Dense(units=8, activation='relu'))
	model.add(Dense(units=1, activation='sigmoid'))
	
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
seed = 7
np.random.seed(seed)
# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
Y = dataset[:, 8]

# 创建模型 for scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# 10折交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print(results.mean())