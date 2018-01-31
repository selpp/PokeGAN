import PIL.Image as Image
import numpy as np
import os

def resize_directory(source, destination, size):
    os.mkdir(destination)
    for filename in os.listdir(source):
        with open(os.path.join(source, filename), 'rb') as file:
            img = Image.open(file)
            img = img.resize(size, Image.BILINEAR)
            img.save(os.path.join(destination, filename), 'JPEG')

def rgba2rgb_directory(source, destination):
    os.mkdir(destination)
    for filename in os.listdir(source):
        with open(os.path.join(source, filename), 'rb') as file:
            img = Image.open(file)

            if img.mode == 'RGBA':
                img.load()
                new_img = Image.new('RGB', img.size, (0, 0, 0))
                new_img.past(img, mask = img.split()[3])
                new_img.save(os.path(destination, filename.split('.')[0] + 'jpg'), 'JPEG')
            else:
                img.convert('RGB')
                img.save(os.path(destination, filename.split('.')[0] + 'jpg'), 'JPEG')

def save(image, path):
    img = Image.fromarray(np.uint8((image / 2.0 + 0.5) * 255.0))
    img.save(path, 'JPEG')
