import os.path
import random
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import tifffile

class AlignedDataset(BaseDataset):
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
        for group in opt.input_list:
            group_info = {'group': group,
                          'conditions': os.listdir(os.path.join(self.root, group))}
            group_info['names'] = os.listdir(os.path.join(self.root, group, group_info['conditions'][0]))
            self.dataset_size += len(group_info['conditions']) * len(group_info['names'])
            self.A_paths.append(group_info)

        # dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        # self.A_paths = sorted(make_dataset(self.dir_A))

        ### train_label B (real images)
        # if not self.exchange and (opt.isTrain or opt.use_encoded_image):
        #     self.B_paths = []
        #     self.B_conds = []
        #     for folder, condition in zip(opt.output_list, opt.output_conditions):
        #         dir_B = os.path.join(opt.dataroot, folder)
        #         paths = make_dataset(dir_B)
        #         self.B_paths += sorted(paths)
        #         self.B_conds += [condition] * len(paths)

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
        ### train_label A (label maps)
        A_group = self.A_paths[index % len(self.A_paths)]
        # Choose an image
        image_name = random.choice(A_group['names'])
        # Choose input and output conditions
        if self.input_condition:
            input_condition = self.input_condition
            output_condition = self.output_condition
        else:
            input_condition, output_condition = random.sample(A_group['conditions'], 2)
        A_path = os.path.join(self.root, A_group['group'], input_condition, image_name)
        # A = tifffile.imread()
        # A = A / 4095
        A = tifffile.imread(A_path)
        # params = get_params(self.opt, A.size)
        # if self.opt.label_nc == 0:
        #     transform_A = get_transform(self.opt, params)
        #     A_tensor = transform_A(A)
        # else:
        #     transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        #     A_tensor = transform_A(A) * 255.0
        A = A / 4095 * 2 - 1
        A_tensor = torch.tensor(A).unsqueeze(0).float()

        B_tensor = inst_tensor = feat_tensor = 0
        ### train_label B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = os.path.join(self.root, A_group['group'], output_condition, image_name)
            B = tifffile.imread(B_path)
            B = B / 4095 * 2 - 1
            B_tensor = torch.tensor(B).unsqueeze(0).float()

            # transform_B = get_transform(self.opt, params)
            # B_tensor = transform_B(B)
            # B = tifffile.imread(os.path.join(self.root, A_group['group'], output_condition, image_name))
            # B = B / 4095
        ### if using instance maps

        # if not self.opt.no_instance:
        #     inst_path = self.inst_paths[index]
        #     inst = Image.open(inst_path)
        #     inst_tensor = transform_A(inst)
        #
        #     if self.opt.load_features:
        #         feat_path = self.feat_paths[index]
        #         feat = Image.open(feat_path).convert('RGB')
        #         norm = normalize()
        #         feat_tensor = norm(transform_A(feat))
        ic_tensor = torch.tensor(float(input_condition)).unsqueeze(0).float()
        oc_tensor = torch.tensor(float(output_condition)).unsqueeze(0).float()

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path, 'input_condition': ic_tensor,
                      'output_condition': oc_tensor}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'