""" Importing Required Libraries """

import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

class SegNet(nn.Module):
    def __init__(self, number_of_classes):
        super(SegNet, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
        self.model_path = load_state_dict_from_url("https://download.pytorch.org/models/vgg13-19584684.pth")
        self.model.load_state_dict(torch.load(self.model_path))
        self.features = list(self.model.features.children())
        self.model.classifier[6].out_features = number_of_classes

    def get_model(self):
        return self.model