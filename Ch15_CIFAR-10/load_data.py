# 导入数据

from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np

(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()

# 展示9张图片样例
for i in range(0, 9):
    plt.subplot(331 + i)
    plt.imshow(toimage(X_train[i]))
plt.show()
