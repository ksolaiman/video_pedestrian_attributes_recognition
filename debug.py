from PIL import Image
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            print(img)
            got_img = True
        except IOError:
            print("das")


read_image("data/mars/bbox_train/0973/0973C1T0002F010.jpg")
