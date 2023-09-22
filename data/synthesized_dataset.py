import os.path
import random
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import tifffile
import numpy as np
import cv2
from scipy.signal import lfilter
from skimage.util import random_noise

class SynthesizedDataset(BaseDataset):

    @staticmethod
    def elementaryCellularAutomata(rule, n, width=None, randfrac=None):
        # Validate inputs
        if not (0 <= rule <= 255):
            raise ValueError("Rule should be an integer between 0 and 255.")

        if not isinstance(n, int) or n <= 0:
            raise ValueError("N should be a positive integer.")

        if width is None:
            width = 2 * n - 1

        if isinstance(width, int):
            if width <= 0:
                raise ValueError("Width should be a positive integer.")
            patt = np.ones(width, dtype=int)
            patt[(width + 1) // 2 - 1] = 2
        else:
            if not all(map(lambda x: x in [0, 1], width)):
                raise ValueError("Width should be a list of 0s and 1s only.")
            patt = np.array(width) + 1
            width = len(patt)

        if randfrac is not None:
            if not (0 <= randfrac <= 1):
                raise ValueError("Randfrac should be a float between 0 and 1.")
            dorand = True
        else:
            dorand = False

        # Unpack rule
        rulearr = [(rule >> i) & 1 for i in range(8)]
        rulearr = np.array(rulearr) + 1

        # Initialize output pattern
        pattern = np.zeros((n, width), dtype=int)

        # Generate the pattern
        for i in range(n):
            pattern[i, :] = patt
            ind = 2 ** 2 * np.roll(patt, 1) + 2 ** 1 * patt + 2 ** 0 * np.roll(patt, -1)
            ind = ind % 8  # Ensure indices are within 0-7
            patt = rulearr[ind]

            # Optional randomization
            if dorand:
                flip = np.random.rand(width) < randfrac
                patt[flip] = 3 - patt[flip]

        # Convert 1 and 2 to 0 and 1
        pattern -= 1

        return pattern

    def caimgen(self, lp_length, imgsize):
        mask = self.elementaryCellularAutomata(np.random.randint(255), lp_length, lp_length, np.random.random())
        pix_value = np.round(np.random.rand(lp_length, lp_length) * 4095).astype(int)
        imgseed = np.array(mask * pix_value, dtype=np.double)

        imgraw = cv2.resize(imgseed, (imgsize * 2, imgsize * 2))  # Double the image size
        imgraw = imgraw[imgsize - 1:2 * imgsize - 1, :]  # Crop the image
        return imgraw

    def blurgen(self, img_gt, k):
        img_gtf = np.zeros_like(img_gt)

        # filter coefficients
        b = [0, k]
        a = [1, -(1 - k)]

        for i in range(img_gt.shape[0]):
            img_gtf[i, :] = lfilter(b, a, img_gt[i, :])
        return img_gtf

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        # self.exchange = opt.condition_exchange
        ### train_label A (label maps)
        # We will see the directory list and the condition list in opt.input_list, opt.input_conditions = []
        self.A_paths = []
        # [[group, conditions, names],]
        self.input_condition = opt.input_condition
        self.output_condition = opt.output_condition
        self.dataset_size = 0
        self.loadSize = opt.loadSize
        self.synthesized = opt.synthesized
        if self.synthesized: # If use synthesized, then input_condition, output_condition are ranges.
            self.dataset_size = opt.dataset_size
            self.input_condition_range = opt.input_condition_range
            self.output_condition_range = opt.output_condition_range
        else:
            for group in opt.input_list:
                group_info = {'group': group,
                              'conditions': os.listdir(os.path.join(self.root, group))}
                group_info['names'] = os.listdir(os.path.join(self.root, group, group_info['conditions'][0]))
                self.dataset_size += len(group_info['conditions']) * len(group_info['names'])
                self.A_paths.append(group_info)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        # self.dataset_size = len(self.A_paths)
      
    def __getitem__(self, index):
        B = inst_tensor = feat_tensor = 0
        if self.synthesized:
            input_condition = random.uniform(self.input_condition_range[0], self.input_condition_range[1])
            output_condition = random.uniform(self.output_condition_range[0], self.output_condition_range[1])
            lp_length = int(32 * 2**random.randint(0, 5))
            img_template = self.caimgen(lp_length, self.loadSize)
            A = img_template + random_noise(img_template, mode='s&p') #random_noise(np.uint16(img_template), mode='speckle')
            A = self.blurgen(A, input_condition)[:, self.loadSize:]
            A = np.uint16(A)
            if self.opt.isTrain or self.opt.use_encoded_image:
                B = img_template + random_noise(img_template, mode='s&p')#np.random.randn(*img_template.shape)
                B = self.blurgen(B, output_condition)[:, self.loadSize:]
                B = np.uint16(B)
            A_path = ''
        else:
            ### train_label A (label maps)
            A_group = self.A_paths[index % len(self.A_paths)]
            # Choose an image
            image_name = random.choice(A_group['names'])
            # # Choose input and output conditions
            # if self.input_condition:
            #     input_condition = self.input_condition
            # else:
            #     input_condition, output_condition = random.sample(A_group['conditions'], 2)
            if self.output_condition:
                output_condition = self.output_condition
                remaining_lst = [val for val in A_group['conditions'] if val != output_condition]
                input_condition = random.choice(remaining_lst)
            else:
                input_condition, output_condition = random.sample(A_group['conditions'], 2)
            A_path = os.path.join(self.root, A_group['group'], input_condition, image_name)
            A = tifffile.imread(A_path)

            ### train_label B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                B_path = os.path.join(self.root, A_group['group'], output_condition, image_name)
                B = tifffile.imread(B_path)

        A = A / 4095 * 2 - 1
        A_tensor = torch.tensor(A).unsqueeze(0).float()
        B = B / 4095 * 2 - 1
        B_tensor = torch.tensor(B).unsqueeze(0).float()
        ic_tensor = torch.tensor(float(input_condition)).unsqueeze(0).float()
        oc_tensor = torch.tensor(float(output_condition)).unsqueeze(0).float()

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path, 'input_condition': ic_tensor,
                      'output_condition': oc_tensor}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'SynthesizedDataset'