# rename image
from PIL import Image
import os
directory = 'map1'

c = 0
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        im = Image.open(directory + '/' + filename)
        name = str(c)+'.jpg'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        c += 1
        print(os.path.join(directory + '/' + 'test', filename))
        continue
    else:
        continue

'''
data = ''
with open('map.csv', 'r') as file:
    data = file.read().replace('\t', '.jpg,')
with open('new_map.csv', 'w') as file:
    file.write(data)

# rename all file in folder 
import pathlib
for path in pathlib.Path("map_detection").iterdir():
    if path.is_file():
        old_name = path.stem
        old_extension = path.suffix
        directory = path.parent
        new_name = "text" + old_name + old_extension
        path.rename(pathlib.Path(directory, new_name))
'''





