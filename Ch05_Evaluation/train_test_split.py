from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')
x = dataset[:, :8]
Y = dataset[:, 8]

# 分割数据集
x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=0.2, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 指定评估数据集
model.fit(x_train, Y_train, validation_data=(x_validation, Y_validation), epochs=150, batch_size=10)