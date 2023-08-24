import os
import tifffile
import cv2
import numpy as np

def data_visualize(root, out_dir):
    for condition in os.listdir(root):
        os.makedirs(os.path.join(out_dir, condition), exist_ok=True)
        for name in os.listdir(os.path.join(root, condition)):
            img = tifffile.imread(os.path.join(root, condition, name))
            img = np.uint8(img / 4095 * 255)
            cv2.imwrite(os.path.join(out_dir, condition, name.split(".")[0] + '.png'), img)

data_visualize("/home/xavier/Documents/Tao-ImageSet/TotalSynthesizedImage/20230703/group1",
               "/home/xavier/Documents/Tao-ImageSet/TotalSynthesizedImage/visualize/group1")