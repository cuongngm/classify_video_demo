# rename image
from PIL import Image
import os
import cv2
import random
from PIL import Image
directory = '/home/cuongngm/Documents/food/food7/'


def convert_file(directory):
    c = 0
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            im = Image.open(directory + filename)
            name = str(c)+'.jpg'
            rgb_im = im.convert('RGB')
            rgb_im.save(directory + 'demo/' + name)
            c += 1
            continue
        else:
            continue


def rename_file(directory):
    for count, filename in enumerate(os.listdir(directory)):
        dst = "food7_" + str(count) + ".jpg"
        src = directory + filename
        dst = directory + dst
        os.rename(src, dst)


def random_file(directory):
    # random file
    size = 400
    for i in range(size):
        randomfile = random.choice([x for x in os.listdir(directory) if os.path.isfile(os.path.join(directory, x))])
        img = Image.open(directory + randomfile).convert("RGB")
        img.save('/home/cuongngm/Documents/video/valid/technology/' + str(i) + ".jpg")
        os.remove(directory + randomfile)


def check_file(directory):
    for file_name in os.listdir(directory):
        img = cv2.imread(directory + "/" + file_name)
        img = cv2.resize(img, (224, 224))


rename_file(directory)


