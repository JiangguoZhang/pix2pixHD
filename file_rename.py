import os
import cv2
import tifffile
import numpy as np

def file_rename(root_dir, out_dir):
    file_list = os.listdir(root_dir)
    while file_list:
        file_name = file_list.pop()
        file_dir = os.path.join(root_dir, file_name)
        if file_name.endswith('tif'):
            img = tifffile.imread(file_dir)
            img = np.uint8(img / 4095 * 255)
            # img = img[:512, :512]
            new_file_name = os.path.join(out_dir, file_name)[:-3] + 'png'
            os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
            cv2.imwrite(new_file_name, img,)
            # new_file_name = file_name.replace("slow", "fast")
            # os.rename(file_dir, os.path.join(root_dir, new_file_name))
        else:
            file_list.extend([os.path.join(file_name, item) for item in os.listdir(file_dir)])
    return 0


file_rename("/home/xavier/Documents/Tao-ImageSet/OneDrive_1_2023-5-22/20230519-worm/worm/in",
            "/home/xavier/Documents/Tao-ImageSet/OneDrive_1_2023-5-22/20230519-worm/worm/in_256")
