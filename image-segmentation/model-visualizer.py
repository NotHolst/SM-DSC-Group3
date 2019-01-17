from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


x_test = np.load('./output/testSet.npy')
x_testMasks = np.load('./output/testSetLabels.npy')

images = 6
startImage = 20

model = load_model('./Models/Model2/model.h5')

res = model.predict(x_test)



def drawImages(page, perPage):
    col = 1
    for index in range(page*perPage, page*perPage + perPage):
        image = res[index]
        plt.subplot(3, perPage, col)
        plt.imshow(image)
        plt.subplot(3, perPage, col+perPage)
        plt.imshow(x_test[index])
        plt.subplot(3, perPage, col+perPage*2)
        plt.imshow(x_testMasks[index])
        col += 1
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')

class PageManager(object):
    page = 0
    perPage = 4

    def next(self, event):
        self.page += 1
        drawImages(self.page, self.perPage)

    def prev(self, event):
        self.page += 1

pageManager = PageManager()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
bprev = Button(axprev, 'Prev')
bprev.on_clicked(pageManager.prev)
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(pageManager.next)

plt.show()
# print(model.summary())