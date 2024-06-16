import torch
import torch.nn as nn
from torchvision.models import vgg19, vgg16
from collections import OrderedDict
from collections import namedtuple
import torchvision.models.vgg as vgg

# VGG 16
vgg_layer = {
    'conv_1_1': 0,
    'relu_1_1': 1,
    'conv_1_2': 2,
    'relu_1_2': 3,
    'pool_1': 4,
    'conv_2_1': 5,
    'relu_2_1': 6,
    'conv_2_2': 7,
    'relu_2_2': 8,
    'pool_2': 9,
    'conv_3_1': 10,
    'relu_3_1': 11,
    'conv_3_2': 12,
    'relu_3_2': 13,
    'conv_3_3': 14,
    'relu_3_3': 15,
    'pool_3': 16,
    'conv_4_1': 17,
    'relu_4_1': 18,
    'conv_4_2': 19,
    'relu_4_2': 20,
    'conv_4_3': 21,
    'relu_4_3': 22,
    'pool_4': 23,
    'conv_5_1': 24,
    'relu_5_1': 25,
    'conv_5_2': 26,
    'relu_5_2': 27,
    'conv_5_3': 28,
    'relu_5_3': 29,
    'pool_5': 30
}

vgg_layer_inv = {
    0: 'conv_1_1',
    1: 'relu_1_1',
    2: 'conv_1_2',
    3: 'relu_1_2',
    4: 'pool_1',
    5: 'conv_2_1',
    6: 'relu_2_1',
    7: 'conv_2_2',
    8: 'relu_2_2',
    9: 'pool_2',
    10: 'conv_3_1',
    11: 'relu_3_1',
    12: 'conv_3_2',
    13: 'relu_3_2',
    14: 'conv_3_3',
    15: 'relu_3_3',
    16: 'pool_3',
    17: 'conv_4_1',
    18: 'relu_4_1',
    19: 'conv_4_2',
    20: 'relu_4_2',
    21: 'conv_4_3',
    22: 'relu_4_3',
    23: 'pool_4',
    24: 'conv_5_1',
    25: 'relu_5_1',
    26: 'conv_5_2',
    27: 'relu_5_2',
    28: 'conv_5_3',
    29: 'relu_5_3',
    30: 'pool_5'
}


class VGG_Model(nn.Module):
    def __init__(self, listen_list=None):
        super(VGG_Model, self).__init__()
        vgg = vgg16(pretrained=True)
        self.vgg_model = vgg.features
        vgg_dict = vgg.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[vgg_layer_inv[index]] = x
        return self.features


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2',
                           'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_4, h_relu4_4, h_relu5_4)

        return out