# rename image
from PIL import Image
import os
import cv2
directory = '/home/cuongngm/Documents/video/babymom/'
'''
c = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpeg"):
        im = Image.open(directory + filename)
        name = str(c)+'.jpg'
        rgb_im = im.convert('RGB')
        rgb_im.save(directory + 'demo/' + name)
        c += 1
        continue
    else:
        continue
'''
for count, filename in enumerate(os.listdir(directory)):
    dst = "news5_" + str(count) + ".jpg"
    src = directory + filename
    dst = directory + dst
    os.rename(src, dst)


