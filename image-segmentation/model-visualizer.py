from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


x_test = np.load('./output/testSet.npy')

images = 6
startImage = 90

model = load_model('./savedModels/model1.h5')

testImages = x_test[startImage:startImage + images]
res = model.predict(testImages)

for index in range(0, len(res)):
    image = res[index]
    plt.subplot(2, images, index+1)
    plt.imshow(image)
    plt.subplot(2, images, index+1 + images)
    plt.imshow(testImages[index])

# plt.show()

print(model.summary())