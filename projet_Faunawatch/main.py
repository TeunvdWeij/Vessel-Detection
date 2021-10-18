import find_text
from PIL import Image
import cv2

from matplotlib import pyplot as plt
from matplotlib import patches


pipeline = find_text.Find_Text()

path = input("Insert the path of image : ")

# Possible to insert an image array
images = [
    cv2.imread(path)
]

prediction_groups = pipeline.recognize(images)

for image in prediction_groups:
    for prediction in image:
        name       = prediction[0]
        coordonnee = prediction[1]
        
        x = coordonnee[0][0]
        y = coordonnee[0][1]
        w = coordonnee[1][0] - x
        h = coordonnee[2][1] - y
        
        im = Image.open(path)

        plt.imshow(im, aspect='auto')
        
        plt.title(name)

        ax = plt.gca()

        rect = patches.Rectangle((x,y),
                         w,
                         h,
                         linewidth=1,
                         edgecolor='cyan',
                         fill = False)

        ax.add_patch(rect)

        plt.show()