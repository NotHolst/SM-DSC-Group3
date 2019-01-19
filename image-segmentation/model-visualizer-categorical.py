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
    "Model -  2 extra layers"
)

models = []
for folder in folders:
    model = load_model('./Models/' + folder + '/model.h5')
    res = model.predict(x_test)
    colorRemap(res)
    models.append(res)

rows = 2 + len(models)
columns = 5

def drawImages(page, perPage):
    imageAxes = []
    col = 1
    firstColumn = True
    for index in range(page*perPage, page*perPage + perPage):
        #Mask and image

        plt.subplot(rows, perPage, col)
        if firstColumn:
            plt.ylabel('Ground Truth', rotation=60, fontsize=8)
        imageAxes.append(plt.imshow(x_testMasks[index]))
        plt.subplot(rows, perPage, col+perPage)
        if firstColumn:
            plt.ylabel('Original image', rotation=60, fontsize=8)
        imageAxes.append(plt.imshow(x_test[index]))

        #Predictions
        for modelIndex in range(2, len(models)+2):
            
           image = models[modelIndex-2][index]
           plt.subplot(rows, perPage, col+perPage*modelIndex)
           if firstColumn:
               yLabelName = folders[modelIndex - 2]
               plt.ylabel(yLabelName, rotation=60, fontsize=8)
           imageAxes.append(plt.imshow(image))
        if firstColumn:
            firstColumn = False
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