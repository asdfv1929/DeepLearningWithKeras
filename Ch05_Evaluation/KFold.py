from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')
x = dataset[:, :8]
Y = dataset[:, 8]

# 分割数据集
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
cvscores = []

for train, validation in kfold.split(x, Y):
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# 训练模型
	model.fit(x[train], Y[train], epochs=150, batch_size=10, verbose=0)
	# 评估模型
	scores = model.evaluate(x[validation], Y[validation], verbose=0)
	# 输出评估结果
	print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
	cvscores.append(scores[1] * 100)
	
# 输出平均值和标准差
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))
