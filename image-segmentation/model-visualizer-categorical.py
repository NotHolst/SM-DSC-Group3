from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
def clamp(n, smallest, largest): return max(smallest, min(n, largest))
import operator


def colorRemap(imageset, hasAlpha = False):
    for mask in imageset:
        for rowIndex in range(0, len(mask)):
            row = mask[rowIndex]
            for pixelIndex in range(0, len(row)):
                pixel = row[pixelIndex]
                
                index = np.argmax(pixel)
                if hasAlpha:
                    mask[rowIndex][pixelIndex] = [0,0,0,0]
                else:
                    mask[rowIndex][pixelIndex] = [0,0,0,255]

                mask[rowIndex][pixelIndex][index] = 255


x_test = np.load('./output/testSet.npy')
x_testMasks = np.load('./output/testSetLabels_categorical.npy')
colorRemap(x_testMasks)

x_test = x_test[0:30]

images = 6
startImage = 20

folders = (
    "Model7 Baseline",
    "Model -  2 extra layers",
    "Model - 2 less layers and half filters",
    "Model - 2 inner Conv2d sigmoid"
)

models = []
for folder in folders:
    model = load_model('./Models/' + folder + '/model.h5')
    res = model.predict(x_test)
    colorRemap(res)
    models.append(res)

#model = load_model('./Models/Model7 Baseline/model.h5')
#res = model.predict(x_test)
#colorRemap(res)

#model2 = load_model('./Models/Model -  2 extra layers/model.h5')
#res2 = model2.predict(x_test)
#colorRemap(res2)

#model3 = load_model('./Models/Model - 2 less layers and half filters/model.h5')
#res3 = model3.predict(x_test)
#colorRemap(res3)

#model4 = load_model('./Models/Model - 2 inner Conv2d sigmoid/model.h5')
#res4 = model4.predict(x_test)
#colorRemap(res4)

rows = 2 + len(models)
columns = 5

def drawImages(page, perPage):
    imageAxes = []
    col = 1
    for index in range(page*perPage, page*perPage + perPage):
        #Mask and image
        plt.subplot(rows, perPage, col)
        imageAxes.append(plt.imshow(x_testMasks[index]))
        plt.subplot(rows, perPage, col+perPage)
        imageAxes.append(plt.imshow(x_test[index]))

        #Predictions
        for modelIndex in range(2, len(models)+2):
           image = models[modelIndex-2][index]
           plt.subplot(rows, perPage, col+perPage*modelIndex)
           imageAxes.append(plt.imshow(image))

        col += 1
    return imageAxes

class PageManager(object):
    page = 0
    perPage = columns

    def __init__(self, fig):
        self.fig = fig
        fig.canvas.mpl_connect('key_press_event', self.keyPress)
        self.pageCount = (len(res)/self.perPage)
        print(self.pageCount)

    def keyPress(self, event):
        if event.key == 'right':
            self.next()
        if event.key == 'left':
            self.prev()

    def next(self):
        self.page += 1
        self.page = int(clamp(self.page, 0, self.pageCount))
        self.updateImages()

    def prev(self):
        self.page -= 1
        self.page = int(clamp(self.page, 0, self.pageCount))
        self.updateImages()
    
    def updateImages(self):
        i = 0
        for index in range(self.page*self.perPage, self.page*self.perPage + self.perPage):
            
            #Mask and image
            imageAxes[i].set_data(x_testMasks[index])
            imageAxes[i+1].set_data(x_test[index])

            #Predictions
            for modelIndex in range(0, len(models)):
                image = models[modelIndex][index]
                imageAxes[i+2+modelIndex].set_data(image)
            i+= rows

        self.fig.canvas.draw()


fig = plt.figure()
imageAxes = drawImages(0, columns)

pageManager = PageManager(fig)
plt.show()
# print(model.summary())