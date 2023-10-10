import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np
from torch_utils.ops import bias_act
from torch_utils import persistence
from torch_utils import misc

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------
@misc.profiled_function
def concatenate_conditions_dynamic(image, conditions):
    # Get the spatial dimensions of the image dynamically
    _, _, height, width = image.shape
    # Expand the conditions to have the same spatial dimensions as the image
    conditions_expanded = conditions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
    # Concatenate along the channel dimension
    concatenated = torch.cat([image, conditions_expanded], dim=1)
    return concatenated

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, cond_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, cond_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, cond_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, cond_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, cond_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, cond_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 mlp_hidden_dim=64, mlp_depth=6, conditional_channels=16):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, cond_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        input_nc += conditional_channels
        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cond_nc = cond_nc
        self.condition_processor = ConditionProcessor(self.cond_nc * 2, mlp_hidden_dim, mlp_depth, conditional_channels)

    def forward(self, input, input_condition, output_condition):
        ### create train_label pyramid
        _, _, height, width = input.shape
        conditions = torch.concatenate([input_condition, output_condition], 1)
        conditions_tensor = self.condition_processor(conditions, height, width)
        input = torch.cat([input, conditions_tensor], dim=1)
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        ### train_img at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, cond_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', mlp_hidden_dim=64, mlp_depth=6, conditional_channels=16):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        
        input_nc += conditional_channels
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)

        self.cond_nc = cond_nc
        self.condition_processor = ConditionProcessor(self.cond_nc * 2, mlp_hidden_dim, mlp_depth, conditional_channels)

    def forward(self, input, input_condition, output_condition):
        _, _, height, width = input.shape
        conditions = torch.concatenate([input_condition, output_condition], 1)
        conditions_tensor = self.condition_processor(conditions, height, width)
        input = torch.cat([input, conditions_tensor], dim=1)
        return self.model(input)
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, cond_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False, mlp_hidden_dim=64, mlp_depth=6, conditional_channels=16):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        input_nc += conditional_channels
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cond_nc = cond_nc
        self.condition_processor = ConditionProcessor(self.cond_nc * 2, mlp_hidden_dim, mlp_depth, conditional_channels)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input, input_condition, output_condition):
        _, _, height, width = input.shape
        conditions = torch.concatenate([input_condition, output_condition], 1)
        conditions_tensor = self.condition_processor(conditions, height, width)
        input = torch.cat([input, conditions_tensor], dim=1)
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ConditionProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=6, output_channels=16, base_height=16, base_width=16):
        super(ConditionProcessor, self).__init__()

        # Part 1: Enhanced MLP
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU()])
            hidden_dim *= 2
        layers.append(nn.Linear(hidden_dim, hidden_dim))  # Final dense representation
        self.enhanced_mlp = nn.Sequential(*layers)

        # Part 2: Dynamic Spatial Broadcast
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, output_channels * base_height * base_width)
        self.base_height = base_height
        self.base_width = base_width
        self.output_channels = output_channels

    def forward(self, x, target_height, target_width):
        # Enhanced MLP processing
        x = self.enhanced_mlp(x)

        # Spatial Broadcasting
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), self.output_channels, self.base_height, self.base_width)

        # Adaptive upsampling to match target dimensions
        x = F.interpolate(x, size=(target_height, target_width), mode='bilinear', align_corners=True)

        return x


# @persistence.persistent_class
# class FullyConnectedLayer(nn.Module):
#     def __init__(self,
#         in_features,                # Number of input features.
#         out_features,               # Number of output features.
#         bias            = True,     # Apply additive bias before the activation function?
#         activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
#         lr_multiplier   = 1,        # Learning rate multiplier.
#         bias_init       = 0,        # Initial value for the additive bias.
#     ):
#         super().__init__()
#         self.activation = activation
#         self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
#         self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
#         self.weight_gain = lr_multiplier / np.sqrt(in_features)
#         self.bias_gain = lr_multiplier
#
#     def forward(self, x):
#         w = self.weight.to(x.dtype) * self.weight_gain
#         b = self.bias
#         if b is not None:
#             b = b.to(x.dtype)
#             if self.bias_gain != 1:
#                 b = b * self.bias_gain
#
#         if self.activation == 'linear' and b is not None:
#             x = torch.addmm(b.unsqueeze(0), x, w.t())
#         else:
#             x = x.matmul(w.t())
#             x = bias_act.bias_act(x, b, act=self.activation)
#         return x
#
#
# @persistence.persistent_class
# class MappingNetwork(nn.Module):
#     def __init__(self,
#         z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
#         c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
#         w_dim,                      # Intermediate latent (W) dimensionality.
#         num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
#         num_layers      = 4,        # Number of mapping layers.
#         embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
#         layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
#         activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
#         lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
#         w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
#     ):
#         super().__init__()
#         self.z_dim = z_dim
#         self.c_dim = c_dim
#         self.w_dim = w_dim
#         self.num_ws = num_ws
#         self.num_layers = num_layers
#         self.w_avg_beta = w_avg_beta
#
#         if embed_features is None:
#             embed_features = w_dim
#         if c_dim == 0:
#             embed_features = 0
#         if layer_features is None:
#             layer_features = w_dim
#         features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
#
#         if c_dim > 0:
#             self.embed = FullyConnectedLayer(c_dim, embed_features)
#         for idx in range(num_layers):
#             in_features = features_list[idx]
#             out_features = features_list[idx + 1]
#             layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
#             setattr(self, f'fc{idx}', layer)
#
#         if num_ws is not None and w_avg_beta is not None:
#             self.register_buffer('w_avg', torch.zeros([w_dim]))
#
#     def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
#         # Embed, normalize, and concat inputs.
#         x = None
#         with torch.autograd.profiler.record_function('input'):
#             if self.z_dim > 0:
#                 misc.assert_shape(z, [None, self.z_dim])
#                 x = normalize_2nd_moment(z.to(torch.float32))
#             if self.c_dim > 0:
#                 misc.assert_shape(c, [None, self.c_dim])
#                 y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
#                 x = torch.cat([x, y], dim=1) if x is not None else y
#
#         # Main layers.
#         for idx in range(self.num_layers):
#             layer = getattr(self, f'fc{idx}')
#             x = layer(x)
#
#         # Update moving average of W.
#         if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
#             with torch.autograd.profiler.record_function('update_w_avg'):
#                 self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
#
#         # Broadcast.
#         if self.num_ws is not None:
#             with torch.autograd.profiler.record_function('broadcast'):
#                 x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
#
#         # Apply truncation.
#         if truncation_psi != 1:
#             with torch.autograd.profiler.record_function('truncate'):
#                 assert self.w_avg_beta is not None
#                 if self.num_ws is None or truncation_cutoff is None:
#                     x = self.w_avg.lerp(x, truncation_psi)
#                 else:
#                     x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
#         return x