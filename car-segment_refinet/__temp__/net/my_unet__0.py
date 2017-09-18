# unet from scratch

from common import *

import torch
import torch.nn as nn
import torch.nn.functional as F

#  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
#  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)


class BCELoss2d_0(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)
# https://stackoverflow.com/questions/45184741/bceloss-for-binary-pixel-wise-segmentation-pytorch
# https://github.com/pytorch/pytorch/issues/751
# for formula: http://geek.csdn.net/news/detail/126833
class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, logit, label):
             neg_abs = - logit.abs()
             loss = logit.clamp(min=0) - logit * label + (1 + neg_abs.exp()).log()
             return loss.mean()

class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        #self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_loss = StableBCELoss()

    def forward(self, logits, targets):
        logits_flat  = logits.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):

        probs = F.sigmoid(logits)
        num = targets.size(0)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


## -------------------------------------------------------------------------------------

def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-4),  #eps=1e-5
        nn.ReLU(inplace=True),
    ]

def make_conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True),
        #nn.ELU(inplace=True),
        #nn.PReLU(out_channels),
    ]

class UNet512_3x3 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_2, self).__init__()
        in_channels, height, width = in_shape

        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(         16, 16, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, padding=1, stride=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, padding=1, stride=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, padding=1, stride=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.50),
        )
        #256

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.50),
        )
        #512

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, padding=0, stride=1 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out    = F.max_pool2d(down1, kernel_size=2, stride=2)

        #256
        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2)

        #128
        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2)

        #64
        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2)

        #32
        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2)

        #16
        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2)

        #8
        out   = self.center(out)
        print(out.size())

        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)

        return out


class UNet512_3x3_residual (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_2, self).__init__()
        in_channels, height, width = in_shape

        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(         16, 16, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, padding=1, stride=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, padding=1, stride=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, padding=1, stride=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
        )
        #512

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, padding=0, stride=1 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out    = F.max_pool2d(down1, kernel_size=2, stride=2)

        #256
        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2)

        #128
        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2)

        #64
        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2)

        #32
        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2)

        #16
        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2)

        #8
        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        #16
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        #32
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        #64
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        #128
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        #256
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        #512
        out   = self.classify(out)

        return out









#pyramid net
class PyNet512   (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_2, self).__init__()
        in_channels, height, width = in_shape

        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(         16, 16, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, padding=1, stride=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, padding=1, stride=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, padding=1, stride=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
        )
        #512

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, padding=0, stride=1 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out    = F.max_pool2d(down1, kernel_size=2, stride=2)

        #256
        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2)

        #128
        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2)

        #64
        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2)

        #32
        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2)

        #16
        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2)

        #8
        out   = self.center(out)
        print(out.size())


        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        #16
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        #32
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        #64
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        #128
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        #256
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        #512
        out   = self.classify(out)

        return out












## 1024 unet ## -------------------------------------------------------------------------
class UNet1024_3x3 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet1024_3x3, self).__init__()
        in_channels, height, width = in_shape

        #1024
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 8, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(          8, 8, kernel_size=3, padding=1, stride=1 ),
        )

        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(  8, 16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu( 16, 16, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.50),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(1024+512,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(512+256,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, padding=1, stride=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(256+128,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, padding=1, stride=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 128+64,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, padding=1, stride=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 64+32,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, padding=1, stride=1 ),
        )
        #256

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 32+16,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, padding=1, stride=1 ),
        )
        #512

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(  16+8,8, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, padding=1, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #1024

        self.classify = nn.Conv2d( 8, num_classes, kernel_size=1, padding=0, stride=1 )


    def forward(self, x):
        #1024
        down0 = self.down0(x)
        out    = F.max_pool2d(down0, kernel_size=2, stride=2)
        #print('down0',out.size())

        #512
        down1 = self.down1(out)
        out    = F.max_pool2d(down1, kernel_size=2, stride=2)
        #print('down1',out.size())

        #256
        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2)
        #print('down2',out.size())

        #128
        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2)
        #print('down3',out.size())

        #64
        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2)
        #print('down4',out.size())

        #32
        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2)
        #print('down5',out.size())

        #16
        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2)
        #print('down6',out.size())     , exit(0)

        #8
        out   = self.center(out)
        #print(out.size())     , exit(0)

        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        #16
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        #32
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        #64
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        #128
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        #256
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        #512
        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        #1024
        out   = torch.squeeze(self.classify(out),1)

        return out









class UNet512_5x5 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_5x5, self).__init__()
        in_channels, height, width = in_shape

        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=5, padding=2, stride=1 ),
        )
        #256

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=5, padding=2, stride=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=5, padding=2, stride=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=5, padding=2, stride=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=5, padding=2, stride=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(512,512, kernel_size=5, padding=2, stride=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=5, padding=2, stride=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=5, padding=2, stride=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=5, padding=2, stride=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=5, padding=2, stride=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=5, padding=2, stride=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=5, padding=2, stride=1 ),
        )
        #256

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=5, padding=2, stride=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=5, padding=2, stride=1 ),
        )
        #512

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out    = F.max_pool2d(down1, kernel_size=2, stride=2)

        #256
        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2)

        #128
        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2)

        #64
        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2)

        #32
        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2)

        #16
        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2)

        #8
        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2)
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)

        return out

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    batch_size  = 4
    C,H,W = 3,1024,2048

    #1024,1024
    #C,H,W = 3,896,640

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = UNet1024_3x3(in_shape=(C,H,W), num_classes=1).cuda().train()
        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        logits = net.forward(x)

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        print(net)

        print('logits')
        print(logits)
    #input('Press ENTER to continue.')