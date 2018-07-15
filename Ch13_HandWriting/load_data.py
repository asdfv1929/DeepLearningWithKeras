# 手写数字识别
# 导入数据

# from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


def load_data(path='mnist.npz'):
	f = np.load(path)
	x_train, y_train = f['x_train'], f['y_train']
	x_test, y_test = f['x_test'], f['y_test']
	f.close()
	return (x_train, y_train), (x_test, y_test)

	
# 从Keras导入Mnist数据集
(X_train, y_train), (X_validation, y_validation) = load_data()

# plt.subplot(221)
# plt.imshow(X_train[0], cmap='gray')
# plt.subplot(222)
# plt.imshow(X_train[1], cmap='gray')
# plt.subplot(223)
# plt.imshow(X_train[2], cmap='gray')
# plt.subplot(224)
# plt.imshow(X_train[3], cmap='gray')
# plt.show()

seed = 7
np.random.seed(seed)

# 多层感知器模型
num_pixels = X_train.shape[1] * X_train.shape[2]
print(num_pixels)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], num_pixels).astype('float32')

# 格式化数据到0~1
X_train = X_train / 255
X_validation = X_validation / 255

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_validation.shape[1]
print(num_classes)

# 定义基准MLP模型
def create_model():
	model = Sequential()
	model.add(Dense(units=num_pixels, activation='relu', input_dim=num_pixels, kernel_initializer='normal'))
	model.add(Dense(units=num_classes, activation='softmax', kernel_initializer='normal'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=200)

score = model.evaluate(X_validation, y_validation)
print('MLP: %.2f%%' % (score[1] * 100))