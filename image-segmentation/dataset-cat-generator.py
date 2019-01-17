from PIL import Image
import numpy as np
import os
import re

'''
[ images
    [ image
        [ row 
            [ 0, 0, 1,  ...n_classes ], [ 0, 0, 1,  ...n_classes ]
        ],
        [ row 
            [ 0, 0, 1,  ...n_classes ], [ 0, 0, 1,  ...n_classes ]
        ]
    ]
]
'''

colorClassMap = dict()
colorClassMap[(255,0,0)] = [1,0,0,0]
colorClassMap[(0,255,0)] = [0,1,0,0]
colorClassMap[(0,0,255)] = [0,0,1,0]
colorClassMap[(0,0,0)]   = [0,0,0,1]


def getpixels(imageFile):
    img = Image.open(imageFile).convert("RGB")
    data = img.getdata()

    output = []

    width, height = data.size
    for y in range(0, width):
        row = []
        for x in range(0, height):
            p = img.getpixel((x, y))
            row.append(colorClassMap[(p[0], p[1], p[2])])
        output.append(row)

    return output

# imageDataArray = [None]
# imageDataArray[0] = getpixels('./testImage.jpg')
# output = np.array(imageDataArray)
# print(output)


def generateDataFile(dirName, startIndex):
    for root, dirs, files in os.walk("./"+dirName):
        imageDataArray = [None] * len(files)
        for filename in files:
            pixels = getpixels(os.path.join(root, filename))
            match = re.search(r"(\d*)", filename)
            index = int(match.group(0)) - startIndex
            imageDataArray[index] = pixels
        np.save("./output/" + dirName+"_categorical", np.array(imageDataArray))


generateDataFile('testSetLabels', 300)
generateDataFile('trainingSetLabels', 0)
