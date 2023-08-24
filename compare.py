import cv2
import os
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import numpy as np
import scipy.stats as stats

def calculate_psnr(img1, img2, max_value=255):
    """Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def compare(input_dir, output_dir, real_dir):
    rmse_before = []
    ssim_before = []
    rmse_after = []
    ssim_after = []
    for img_name in os.listdir(input_dir):
        img_name_ = img_name.split(".")[0]
        img1 = cv2.imread(os.path.join(input_dir, img_name), cv2.IMREAD_GRAYSCALE) / 255
        img2 = cv2.imread(os.path.join(output_dir, "%s_synthesized_image.jpg" % img_name_), cv2.IMREAD_GRAYSCALE) / 255
        img3 = cv2.imread(os.path.join(real_dir, img_name.replace("fast", "slow")), cv2.IMREAD_GRAYSCALE) / 255
        rmse_before.append(np.sqrt(np.mean(np.square(img3 - img1))))
        rmse_after.append(np.sqrt(np.mean(np.square(img3 - img2))))
        ssim_before.append(ssim(img3, img1))
        ssim_after.append(ssim(img3, img2))
    print("%f\t%f" % (np.mean(rmse_before), np.std(rmse_before)))
    print("%f\t%f" % (np.mean(rmse_after), np.std(rmse_after)))
    print("%f\t%f" % (np.mean(ssim_before), np.std(ssim_before)))
    print("%f\t%f" % (np.mean(ssim_after), np.std(ssim_after)))
    # for i in range(len(rmse_before)):
    #     print("%d\t%.3f\t%.3f\t%.3f\t%.3f" % (i, rmse_before[i], rmse_after[i], ssim_before[i], ssim_after[i]))
    print(stats.ttest_rel(rmse_before, rmse_after))
    print(stats.ttest_rel(ssim_before, ssim_after))
    return 0


compare("/home/xavier/PycharmProjects/pix2pixHD/datasets/liver/test_A",
        "/home/xavier/PycharmProjects/pix2pixHD/results/liver_1024p/test_latest/images",
        "/home/xavier/PycharmProjects/pix2pixHD/datasets/liver/test_B")