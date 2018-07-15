from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
x = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
# validation_split设置验证集的百分比
model.fit(x=x, y=Y, epochs=150, batch_size=10, validation_split=0.2)
