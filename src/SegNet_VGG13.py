""" Importing Required Libraries """

import torch
import torch.nn as nn
import torch.nn.functional as F

""" This class forms the entire SegNet model including the required Convolution Layers, Pooling-Unpooling Layers and 
Normalization along with ReLU Activation Function. """
class SegNet(nn.Module):

    def __init__(self, number_of_classes):
        super(SegNet, self).__init__()
        self.momentum = 0.9

        """ Encoder: 13 Convolution Layers as per VGG architecture with Batch Normalization per layer. """
        self.conv_enc_11 = nn.Conv2d(3, 64, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_11 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.conv_enc_12 = nn.Conv2d(64, 64, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_12 = nn.BatchNorm2d(64, momentum=self.momentum)

        self.conv_enc_21 = nn.Conv2d(64, 128, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_21 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.conv_enc_22 = nn.Conv2d(128, 128, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_22 = nn.BatchNorm2d(128, momentum=self.momentum)

        self.conv_enc_31 = nn.Conv2d(128, 256, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_31 = nn.BatchNorm2d(256, momentum=self.momentum)
        self.conv_enc_32 = nn.Conv2d(256, 256, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_32 = nn.BatchNorm2d(256, momentum=self.momentum)
        self.conv_enc_33 = nn.Conv2d(256, 256, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_33 = nn.BatchNorm2d(256, momentum=self.momentum)

        self.conv_enc_41 = nn.Conv2d(256, 512, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_41 = nn.BatchNorm2d(512, momentum=self.momentum)
        self.conv_enc_42 = nn.Conv2d(512, 512, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_42 = nn.BatchNorm2d(512, momentum=self.momentum)
        self.conv_enc_43 = nn.Conv2d(512, 512, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_43 = nn.BatchNorm2d(512, momentum=self.momentum)

        self.conv_enc_51 = nn.Conv2d(512, 1024, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_51 = nn.BatchNorm2d(1024, momentum=self.momentum)
        self.conv_enc_52 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_52 = nn.BatchNorm2d(1024, momentum=self.momentum)
        self.conv_enc_53 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same', stride=1)
        self.batch_norm_enc_53 = nn.BatchNorm2d(1024, momentum=self.momentum)

        """ Decoder: 13 Convolution Layers as per the Encoder with Batch Normalization per layer. """
        self.conv_dec_11 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_11 = nn.BatchNorm2d(1024, momentum=self.momentum)
        self.conv_dec_12 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_12 = nn.BatchNorm2d(1024, momentum=self.momentum)
        self.conv_dec_13 = nn.Conv2d(1024, 512, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_13 = nn.BatchNorm2d(512, momentum=self.momentum)

        self.conv_dec_21 = nn.Conv2d(512, 512, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_21 = nn.BatchNorm2d(512, momentum=self.momentum)
        self.conv_dec_22 = nn.Conv2d(512, 512, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_22 = nn.BatchNorm2d(512, momentum=self.momentum)
        self.conv_dec_23 = nn.Conv2d(512, 256, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_23 = nn.BatchNorm2d(256, momentum=self.momentum)

        self.conv_dec_31 = nn.Conv2d(256, 256, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_31 = nn.BatchNorm2d(256, momentum=self.momentum)
        self.conv_dec_32 = nn.Conv2d(256, 256, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_32 = nn.BatchNorm2d(256, momentum=self.momentum)
        self.conv_dec_33 = nn.Conv2d(256, 128, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_33 = nn.BatchNorm2d(128, momentum=self.momentum)

        self.conv_dec_41 = nn.Conv2d(128, 128, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_41 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.conv_dec_42 = nn.Conv2d(128, 64, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_42 = nn.BatchNorm2d(64, momentum=self.momentum)

        self.conv_dec_51 = nn.Conv2d(64, 64, kernel_size=3, padding='same', stride=1)
        self.batch_norm_dec_51 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.conv_dec_52 = nn.Conv2d(64, number_of_classes, kernel_size=3, padding='same', stride=1)

    """ This method forms the actual sequential network. This plugs in the actual layers by passing the required input 
   and processing it accordingly. """
    def forward(self, input):

        """ Encoder Layers take in the convolved image and process through relu layer before passing onto next
        convolution layer. These are then max-pooled to store the important indices which will be used in decoder for
        up-sampling. """
        input_enc_11 = F.relu(self.batch_norm_enc_11(self.conv_enc_11(input)))
        input_enc_12 = F.relu(self.batch_norm_enc_12(self.conv_enc_12(input_enc_11)))
        input_enc_pooled_1, id_enc_pooled_1 = F.max_pool2d(input_enc_12, kernel_size=2, stride=2, return_indices=True)

        input_enc_21 = F.relu(self.batch_norm_enc_21(self.conv_enc_21(input_enc_pooled_1)))
        input_enc_22 = F.relu(self.batch_norm_enc_22(self.conv_enc_22(input_enc_21)))
        input_enc_pooled_2, id_enc_pooled_2 = F.max_pool2d(input_enc_22, kernel_size=2, stride=2, return_indices=True)

        input_enc_31 = F.relu(self.batch_norm_enc_31(self.conv_enc_31(input_enc_pooled_2)))
        input_enc_32 = F.relu(self.batch_norm_enc_32(self.conv_enc_32(input_enc_31)))
        input_enc_33 = F.relu(self.batch_norm_enc_33(self.conv_enc_33(input_enc_32)))
        input_enc_pooled_3, id_enc_pooled_3 = F.max_pool2d(input_enc_33, kernel_size=2, stride=2, return_indices=True)

        input_enc_41 = F.relu(self.batch_norm_enc_41(self.conv_enc_41(input_enc_pooled_3)))
        input_enc_42 = F.relu(self.batch_norm_enc_42(self.conv_enc_42(input_enc_41)))
        input_enc_43 = F.relu(self.batch_norm_enc_43(self.conv_enc_43(input_enc_42)))
        input_enc_pooled_4, id_enc_pooled_4 = F.max_pool2d(input_enc_43, kernel_size=2, stride=2, return_indices=True)

        input_enc_51 = F.relu(self.batch_norm_enc_51(self.conv_enc_51(input_enc_pooled_4)))
        input_enc_52 = F.relu(self.batch_norm_enc_52(self.conv_enc_52(input_enc_51)))
        input_enc_53 = F.relu(self.batch_norm_enc_53(self.conv_enc_53(input_enc_52)))
        input_enc_pooled_5, id_enc_pooled_5 = F.max_pool2d(input_enc_53, kernel_size=2, stride=2, return_indices=True)

        """ Decoder Laters take in the encoded image, uses the pooled indices to un-pool and passes it to the following 
        convolution layers. This finally produces the output image. """
        input_dec_unpooled_1 = F.max_unpool2d(input_enc_pooled_5, id_enc_pooled_5, kernel_size=2, stride=2)
        input_dec_11 = F.relu(self.batch_norm_dec_11(self.conv_dec_11(input_dec_unpooled_1)))
        input_dec_12 = F.relu(self.batch_norm_dec_12(self.conv_dec_12(input_dec_11)))
        input_dec_13 = F.relu(self.batch_norm_dec_13(self.conv_dec_13(input_dec_12)))

        input_dec_unpooled_2 = F.max_unpool2d(input_dec_13, id_enc_pooled_4, kernel_size=2, stride=2)
        input_dec_21 = F.relu(self.batch_norm_dec_21(self.conv_dec_21(input_dec_unpooled_2)))
        input_dec_22 = F.relu(self.batch_norm_dec_22(self.conv_dec_22(input_dec_21)))
        input_dec_23 = F.relu(self.batch_norm_dec_23(self.conv_dec_23(input_dec_22)))

        input_dec_unpooled_3 = F.max_unpool2d(input_dec_23, id_enc_pooled_3, kernel_size=2, stride=2)
        input_dec_31 = F.relu(self.batch_norm_dec_31(self.conv_dec_31(input_dec_unpooled_3)))
        input_dec_32 = F.relu(self.batch_norm_dec_32(self.conv_dec_32(input_dec_31)))
        input_dec_33 = F.relu(self.batch_norm_dec_33(self.conv_dec_33(input_dec_32)))

        input_dec_unpooled_4 = F.max_unpool2d(input_dec_33, id_enc_pooled_2, kernel_size=2, stride=2)
        input_dec_41 = F.relu(self.batch_norm_dec_41(self.conv_dec_41(input_dec_unpooled_4)))
        input_dec_42 = F.relu(self.batch_norm_dec_42(self.conv_dec_42(input_dec_41)))

        input_dec_unpooled_5 = F.max_unpool2d(input_dec_42, id_enc_pooled_1, kernel_size=2, stride=2)
        input_dec_51 = F.relu(self.batch_norm_dec_51(self.conv_dec_51(input_dec_unpooled_5)))
        input_dec_52 = self.conv_dec_52(input_dec_51)

        return input_dec_52