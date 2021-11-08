from sklearn.datasets import load_digits

digits = load_digits()

#print(digits.DESCR)

#print(digits.data[150])
#print(digits.target[150])

#print(digits.data[5])
#print(digits.target[5])

#print(digits.data.shape)
#print(digits.target.shape)

import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4))

#plt.show()

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(digits.data, digits.target, random_state=11)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

#43:56