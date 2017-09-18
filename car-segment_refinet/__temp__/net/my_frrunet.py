# unet from scratch
from common import *
from net.segmentation.loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4  #1e-4  #1e-5


#-------------------------------------------------------------------------------------------
class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

        if is_bn   is False: self.bn  =None
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x

    def merge_bn_to_conv(self):
        #todo ...
        pass

## -----------------------------------------------------------------------------------------------------------

## 'Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes'

# z: prediction mask
# y: features
class FRRU (nn.Module):
    def __init__(self, y_previous_channels, z_previous_channels, y_channels, z_channels):
        super(FRRU, self).__init__()
        assert(z_previous_channels==z_channels)

        self.conv0 = ConvBnRelu2d(z_previous_channels+y_previous_channels, y_channels, kernel_size=3, padding=1,  stride=1)
        self.conv1 = ConvBnRelu2d(y_channels, y_channels, kernel_size=3, padding=1,  stride=1 )
        self.residual = nn.Conv2d(y_channels, z_channels, kernel_size=1, padding=1,  stride=1,  bias=True)

    def forward(self, y_previous, z_previous):
        y = torch.cat([y_previous,y],1)
        y = self.conv0(y)
        y = self.conv1(y)
        z = self.residual(y)
        return  y, z


class DownFRRU (FRRU):
     def forward(self, y_previous, z_previous):
        y, i = F.max_pool2d(y_previous, kernel_size=2, stride=2, return_indices=True)
        H, W = y.size(-1),y.size(-2)
        z = F.upsample(z_previous,size=(H,W),mode='bilinear')
        y = torch.cat([z,y],1)
        y = self.conv0(y)
        y = self.conv1(y)

        H, W = z_previous.size(-1),z_previous.size(-2)
        z = self.residual(y)
        z = F.upsample(z,size=(H,W),mode='bilinear')
        z = z+z_previous

        return  y, z, i


class UpFRRU(FRRU):
     def forward(self, y_previous, z_previous, index):
        y    = F.max_unpool2d(y_previous, index, kernel_size=2, stride=2)
        H, W = y.size(-1),y.size(-2)
        z = F.upsample(z_previous,size=(H,W),mode='bilinear')
        y = torch.cat([z,y],1)
        y = self.conv0(y)
        y = self.conv1(y)

        H, W = z_previous.size(-1),z_previous.size(-2)
        z = self.residual(y)
        z = F.upsample(z,size=(H,W),mode='bilinear')
        z = z+z_previous

        return  y, z


#ResidualBlock
class RU (nn.Module):
    def __init__(self, z_previous_channels, z_channels, stride=1):
        super(RU, self).__init__()

        self.block = nn.Sequential(
            ConvBnRelu2d(z_previous_channels, z_channels, kernel_size=3, padding=1,  stride=1 ),
            ConvBnRelu2d(z_channels,          z_channels, kernel_size=3, padding=1,  stride=1, is_relu=False),
        )
        #self.shortcut = None
        #if z_previous_channels!=z_channels or stride!=1:
        #    self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride,  bias=True)

    def forward(self, z_previous):
        z = self.block(z_previous)
        #z_previous = z_previous if self.shortcut is None else self.shortcut(z_previous)
        z = F.relu(z+z_previous, inplace=True)
        return z









#############################################################################################
class FRRUNet512_0(nn.Module):
    def __init__(self, in_shape):
        super(FRRUNet512, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)


        #512
        self.preprocess = nn.Sequential(
            ConvBnRelu2d(3,32,kernel_size=5,padding=2, stride=1),
            RU(32,32),
            #RU(32,32),
        )
        # y_previous_channels, z_previous_channels, y_channels, z_channels
        #512
        self.down1 = DownFRRU( 32, 32,  32, 32)   #256
        self.down2 = DownFRRU( 32, 32,  32, 32)   #128
        self.down3 = DownFRRU( 32, 32,  64, 32)   # 64
        self.down4 = DownFRRU( 64, 32, 128, 32)   # 32
        self.down5 = DownFRRU(128, 32, 256, 32)   # 16
        self.down6 = DownFRRU(256, 32, 512, 32)   #  8

        self.center= ConvBnRelu2d(512, 256, kernel_size=3, padding=1, dilation=1, stride=1, groups=1)

        # 8
        self.up6 = UpFRRU(256, 32, 128, 32) # 16
        self.up5 = UpFRRU(128, 32,  64, 32) # 32
        self.up4 = UpFRRU( 64, 32,  32, 32) # 64
        self.up3 = UpFRRU( 32, 32,  32, 32) #128
        self.up2 = UpFRRU( 32, 32,  32, 32) #256
        self.up1 = UpFRRU( 32, 32,  32, 32) #512

        self.postprocess = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1, bias=True),
            RU(32,32),
            #RU(32,32),
        )
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        #512
        y = F.upsample(x,size=(512, 512),mode='bilinear')
        y = self.preprocess(y)
        z = y
        #[1, 32, 512, 512]

        #down ---
        # z: prediction mask
        # y: features
        y, z, i1  = self.down1(y, z)  # y: 256
        y, z, i2  = self.down2(y, z)  # y: 128
        y, z, i3  = self.down3(y, z)  # y:  64
        y, z, i4  = self.down4(y, z)  # y:  32
        y, z, i5  = self.down5(y, z)  # y:  16
        y, z, i6  = self.down6(y, z)  # y:   8

        y = self.center(y)

        # up ---
        y, z = self.up6(y, z, i6)  # y: 16
        y, z = self.up5(y, z, i5)  # y: 32
        y, z = self.up4(y, z, i4)  # y: 64
        y, z = self.up3(y, z, i3)  # y:128
        y, z = self.up2(y, z, i2)  # y:256
        y, z = self.up1(y, z, i1)  # y:512

        z = torch.cat([z,y],1)
        z = self.postprocess(z)
        z = self.classify(z)
        z = F.upsample(z,size=(self.H, self.W),mode='bilinear')
        z = torch.squeeze(z, dim=1)
        return z


class FRRUNet512(nn.Module):
    def __init__(self, in_shape):
        super(FRRUNet512, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)


        #512
        self.preprocess = nn.Sequential(
            ConvBnRelu2d(3,32,kernel_size=5,padding=2, stride=1),
            #RU(32,32),
            #RU(32,32),
        )
        # y_previous_channels, z_previous_channels, y_channels, z_channels
        #512
        self.down1 = DownFRRU( 32, 32,  32, 32)   #256
        self.down2 = DownFRRU( 32, 32,  64, 32)   #128
        self.down3 = DownFRRU( 64, 32, 128, 32)   # 64
        self.down4 = DownFRRU(128, 32, 256, 32)   # 32
        self.down5 = DownFRRU(256, 32, 512, 32)   # 16
        self.down6 = DownFRRU(512, 32, 512, 32)   #  8

        #self.center= ConvBnRelu2d(512, 256, kernel_size=3, padding=1, dilation=1, stride=1, groups=1)
        self.center = None
        
        # 8
        self.up6 = UpFRRU(512, 32, 256, 32) # 16
        self.up5 = UpFRRU(256, 32, 128, 32) # 32
        self.up4 = UpFRRU(128, 32,  64, 32) # 64
        self.up3 = UpFRRU( 64, 32,  32, 32) #128
        self.up2 = UpFRRU( 32, 32,  32, 32) #256
        self.up1 = UpFRRU( 32, 32,  32, 32) #512

        self.postprocess = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1, bias=True),
            #RU(32,32),
            #RU(32,32),
        )
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        #512
        y = F.upsample(x,size=(512, 512),mode='bilinear')
        y = self.preprocess(y)
        z = y
        #[1, 32, 512, 512]

        #down ---
        # z: prediction mask
        # y: features
        y, z, i1  = self.down1(y, z)  # y: 256
        y, z, i2  = self.down2(y, z)  # y: 128
        y, z, i3  = self.down3(y, z)  # y:  64
        y, z, i4  = self.down4(y, z)  # y:  32
        y, z, i5  = self.down5(y, z)  # y:  16
        y, z, i6  = self.down6(y, z)  # y:   8

        if self.center is not None:
            y = self.center(y)

        # up ---
        y, z = self.up6(y, z, i6)  # y: 16
        y, z = self.up5(y, z, i5)  # y: 32
        y, z = self.up4(y, z, i4)  # y: 64
        y, z = self.up3(y, z, i3)  # y:128
        y, z = self.up2(y, z, i2)  # y:256
        y, z = self.up1(y, z, i1)  # y:512

        z = torch.cat([z,y],1)
        z = self.postprocess(z)
        z = self.classify(z)
        z = F.upsample(z,size=(self.H, self.W),mode='bilinear')
        z = torch.squeeze(z, dim=1)
        return z


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 1
    C,H,W = 3,CARVANA_HEIGHT,CARVANA_WIDTH

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = FRRUNet512(in_shape=(C,H,W)).cuda().train()
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


