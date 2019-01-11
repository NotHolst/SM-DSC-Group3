from PIL import Image
import numpy as np
import os, re


def getpixels(imageFile):
    img = Image.open("./testSet/0.jpg").convert("RGB")
    data = img.getdata()

    red = []
    blue = []
    green = []

    width, height = data.size
    for y in range(0, width):
        for x in range(0, height):
            p = img.getpixel((x, y))
            red.append(p[0])
            green.append(p[1])
            blue.append(p[2])

    return red + green + blue


for root, dirs, files in os.walk("./testSet"):
    imageDataArray = [None] * len(files)
    for filename in files:
        pixels = getpixels(filename)
        match = re.search("(\d*)", filename)
        index = int(match.group(0))
        imageDataArray[index] = pixels
    output = np.array(imageDataArray)
    np.save("./output/testSet", output)

for root, dirs, files in os.walk("./trainingSet"):
    imageDataArray = [None] * len(files)
    for filename in files:
        pixels = getpixels(filename)
        match = re.search("(\d*)", filename)
        index = int(match.group(0))
        imageDataArray[index] = pixels
    output = np.array(imageDataArray)
    np.save("./output/trainingSet", output)

