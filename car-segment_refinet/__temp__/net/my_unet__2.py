# unet from scratch
from common import *
from net.segmentation.loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_EPS = 1e-4  #1e-4  #1e-5

def make_norm_channel(x):
    z = torch.unsqueeze(torch.sum(x,dim=1),1)
    z2_mean = F.avg_pool2d(z*z,kernel_size=7,padding=3,stride=1)
    z_mean  = F.avg_pool2d(z,  kernel_size=7,padding=3,stride=1)
    z_std   = torch.sqrt(z2_mean-z_mean*z_mean+0.03)
    z = (z-z_mean)/(z_std)
    return z

def make_location_channel(z):
    B,C,H,W = z.size()
    x = torch.arange(0, W).float().cuda()/W
    y = torch.arange(0, H).float().cuda()/H
    y = y.view(H,1)#.contiguous()
    yy = y.repeat(B,1,1,W)
    xx = x.repeat(B,1,H,1)

    return yy,xx


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


class ConvResidual (nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvResidual, self).__init__()

        self.block = nn.Sequential(
            ConvBnRelu2d(in_channels,  out_channels, kernel_size=3, padding=1,  stride=1 ),
            ConvBnRelu2d(out_channels, out_channels, kernel_size=3, padding=1,  stride=1, is_relu=False),
        )
        self.shortcut = None
        if in_channels!=out_channels or stride!=1:
           self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride,  bias=True)

    def forward(self, x):
        r = x if self.shortcut is None else self.shortcut(x)
        x = self.block(x)
        x = F.relu(x+r, inplace=True)
        return x



## -----------------------------------------------------------------------------------------------------------

## origainl 3x3 stack filters used in UNet
class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self, x_big, x):
        #y = F.upsample(x, size=(x_big.size(-2),x_big.size(-1)),mode='bilinear')
        y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y
## -----------------------------------------------------------------------------------------------------------

## use upooling
class PoolStackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(PoolStackEncoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small, index  = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        return y, y_small, index


class PoolStackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(PoolStackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self, x_big, x, index):
        y = F.max_unpool2d(x, index, kernel_size=2, stride=2 )
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y









#
# ## -----------------------------------------------------------------------------------------------------------
# class DeepStackEncoder (nn.Module):
#     def __init__(self, x_channels, y_channels, kernel_size=3):
#         super(DeepStackEncoder, self).__init__()
#         padding=(kernel_size-1)//2
#         self.encode = nn.Sequential(
#             ConvBnRelu2d(x_channels, y_channels//2, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
#             ConvBnRelu2d(y_channels//2, y_channels//2, kernel_size=1, padding=0, dilation=1, stride=1, groups=1),
#             ConvBnRelu2d(y_channels//2, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
#         )
#
#     def forward(self,x):
#         y = self.encode(x)
#         y_small = F.max_pool2d(y, kernel_size=2, stride=2)
#         return y, y_small
#
#
# class DeepStackDecoder (nn.Module):
#     def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
#         super(DeepStackDecoder, self).__init__()
#         padding=(kernel_size-1)//2
#
#         self.decode = nn.Sequential(
#             ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
#             ConvBnRelu2d(y_channels,    y_channels//2, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
#             ConvBnRelu2d(y_channels//2, y_channels//2, kernel_size=1, padding=0, dilation=1, stride=1, groups=1),
#             ConvBnRelu2d(y_channels//2, y_channels,    kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
#         )
#
#     def forward(self, x_big, x):
#         #y = F.upsample(x, size=(x_big.size(-2),x_big.size(-1)),mode='bilinear')
#         y = F.upsample(x, scale_factor=2,mode='bilinear')
#         y = torch.cat([y,x_big],1)
#         y = self.decode(y)
#         return  y



## -----------------------------------------------------------------------------------------------------------
class DeepStackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(DeepStackEncoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvResidual(y_channels, y_channels),
            #ConvResidual(y_channels, y_channels),
            #ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class DeepStackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(DeepStackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            #ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvResidual(y_channels, y_channels),
            ConvResidual(y_channels, y_channels),
            #ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self, x_big, x):
        #y = F.upsample(x, size=(x_big.size(-2),x_big.size(-1)),mode='bilinear')
        y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y
#############################################################################################

class UNet1024_0(nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)

        #1024
        #self.down0 = StackEncoder(  3, 16)   #512
        self.down1 = StackEncoder(  4, 16)   #256
        self.down2 = StackEncoder( 16, 32)   #128
        self.down3 = StackEncoder( 32, 64)   # 64
        self.down4 = StackEncoder( 64,128)   # 32
        self.down5 = StackEncoder(128,256)   # 16
        self.down6 = StackEncoder(256,512)   #  8

        self.center = nn.Sequential(
            ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        self.up6 = StackDecoder(512,1024, 512) # 16
        self.up5 = StackDecoder(256, 512, 256) # 32
        self.up4 = StackDecoder(128, 256, 128) # 64
        self.up3 = StackDecoder( 64, 128,  64) #128
        self.up2 = StackDecoder( 32,  64,  32) #256
        self.up1 = StackDecoder( 16,  32,  16) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.classify1 = nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        #1024
        out = F.upsample(x,size=(512, 512),mode='bilinear')
        if 1:
            z = make_norm_channel(out)
            out = torch.cat([z,out],1)

        #down0,out = self.down0(out)  #512
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center(out)

        #debug
        print('x    ',x.size())
        print('down1',down1.size())
        print('down2',down2.size())
        print('down3',down3.size())
        print('down4',down4.size())
        print('down5',down5.size())
        print('down6',down6.size())
        print('out  ',out.size())     #, exit(0)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #out = self.up0(down0, out)
        #1024

        out = self.classify1(out)
        out = F.upsample(out,size=(self.H, self.W),mode='bilinear')
        out = torch.squeeze(out, dim=1)
        return out



class UNet1024 (nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)
        self.C = 4 #add contrast channel

        #1024
        #self.down0 = StackEncoder(  3, 16)    #512
        self.down1 = StackEncoder(self.C,32, kernel_size=5)   #256
        self.down2 = StackEncoder( 32,   64, kernel_size=5)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3) # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3) # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3) # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3) #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=5) #256
        self.up1 = StackDecoder(  32,  32,  32, kernel_size=5) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        #1024
        out = F.upsample(x,size=(512, 512),mode='bilinear')
        if self.C==4:
            z = make_norm_channel(out)
            out = torch.cat([z,out],1)

        #down0,out = self.down0(out)  #512
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center(out)

        #debug
        if 0:
            print('x    ',x.size())
            print('down1',down1.size())
            print('down2',down2.size())
            print('down3',down3.size())
            print('down4',down4.size())
            print('down5',down5.size())
            print('down6',down6.size())
            print('out  ',out.size())     #, exit(0)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #out = self.up0(down0, out)
        #1024

        out = self.classify(out)
        out = F.upsample(out,size=(self.H, self.W),mode='bilinear')
        out = torch.squeeze(out, dim=1)
        return out




class RefineUNet1024 (nn.Module):
    def __init__(self, in_shape):
        super(RefineUNet1024, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)
        self.C = 4 #add contrast channel

        #1024
        #self.down0 = StackEncoder(  3, 16)    #512
        self.down1 = StackEncoder(self.C,32, kernel_size=5)   #256
        self.down2 = StackEncoder( 32,   64, kernel_size=3)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3) # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3) # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3) # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3) #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3) #256
        self.up1 = StackDecoder(  32,  32,  32, kernel_size=5) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.refine = nn.Sequential(
            ConvBnRelu2d(4+32, 32, kernel_size=5, padding=2, stride=1 ),
            ConvBnRelu2d(  32, 32, kernel_size=5, padding=2, stride=1 ),
        )
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        #1024
        out = F.upsample(x,size=(512, 512),mode='bilinear')
        if self.C==4:
            z = make_norm_channel(out)
            out = torch.cat([z,out],1)

        #down0,out = self.down0(out)  #512
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center(out)

        #debug
        if 0:
            print('x    ',x.size())
            print('down1',down1.size())
            print('down2',down2.size())
            print('down3',down3.size())
            print('down4',down4.size())
            print('down5',down5.size())
            print('down6',down6.size())
            print('out  ',out.size())     #, exit(0)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #out = self.up0(down0, out)
        #1024


        # refinement
        out = F.upsample(out,size=(self.H//2, self.W//2), mode='bilinear')
        x   = F.upsample(x,  size=(self.H//2, self.W//2), mode='bilinear')
        z   = make_norm_channel(x)
        out = torch.cat([x, z, out],1)
        out = self.refine(out)

        out = self.classify(out)
        out = F.upsample(out,size=(self.H, self.W),mode='bilinear')
        out = torch.squeeze(out, dim=1)
        return out


# stand Unet without upsize
class UNet (nn.Module):
    def __init__(self, in_shape):
        super(UNet, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)


        #1024
        #self.down0 = StackEncoder(  3, 16)    #512
        self.down1 = StackEncoder(self.C,32, kernel_size=3)   #256
        self.down2 = StackEncoder( 32,   64, kernel_size=3)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3) # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3) # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3) # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3) #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3) #256
        self.up1 = StackDecoder(  32,  32,  32, kernel_size=3) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        #1024
        out = x

        #down0,out = self.down0(out) #512
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center(out)

        #debug
        if 0:
            print('x    ',x.size())
            print('down1',down1.size())
            print('down2',down2.size())
            print('down3',down3.size())
            print('down4',down4.size())
            print('down5',down5.size())
            print('down6',down6.size())
            print('out  ',out.size())     #, exit(0)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #out = self.up0(down0, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out



# stand Unet without upsize
class MoreUNet (nn.Module):
    def __init__(self, in_shape):
        super(MoreUNet, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)



        #1024
        #self.down0 = StackEncoder(  3, 16)    #512
        self.down1 = DeepStackEncoder(self.C, 32, kernel_size=3)   #256
        self.down2 = DeepStackEncoder(  32,   64, kernel_size=3)   #128
        self.down3 = DeepStackEncoder(  64,  128, kernel_size=3)   # 64
        self.down4 = DeepStackEncoder( 128,  256, kernel_size=3)   # 32
        self.down5 = DeepStackEncoder( 256,  512, kernel_size=3)   # 16
        self.down6 = DeepStackEncoder( 512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvResidual(1024, 1024),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = DeepStackDecoder(1024, 1024,  512, kernel_size=3) # 16
        self.up5 = DeepStackDecoder( 512,  512,  256, kernel_size=3) # 32
        self.up4 = DeepStackDecoder( 256,  256,  128, kernel_size=3) # 64
        self.up3 = DeepStackDecoder( 128,  128,   64, kernel_size=3) #128
        self.up2 = DeepStackDecoder(  64,   64,   32, kernel_size=3) #256
        self.up1 = DeepStackDecoder(  32,   32,   32, kernel_size=3) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self,x):

        #1024
        out = x
        if 0:
            xx, yy = make_location_channel(x)
            out = torch.cat((out,xx,yy),1)



        #down0,out = self.down0(out) #512
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center(out)

        #debug
        if 0:
            print('x    ',x.size())
            print('down1',down1.size())
            print('down2',down2.size())
            print('down3',down3.size())
            print('down4',down4.size())
            print('down5',down5.size())
            print('down6',down6.size())
            print('out  ',out.size())     #, exit(0)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #out = self.up0(down0, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out




# stand Unet without upsize
class PoolUNet (nn.Module):
    def __init__(self, in_shape):
        super(PoolUNet, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)



        #1024
        #self.down0 = StackEncoder(  3, 16)    #512
        self.down1 = PoolStackEncoder(self.C, 32, kernel_size=3)   #256
        self.down2 = PoolStackEncoder(  32,   64, kernel_size=3)   #128
        self.down3 = PoolStackEncoder(  64,  128, kernel_size=3)   # 64
        self.down4 = PoolStackEncoder( 128,  256, kernel_size=3)   # 32
        self.down5 = PoolStackEncoder( 256,  512, kernel_size=3)   # 16
        self.down6 = PoolStackEncoder( 512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = PoolStackDecoder(1024, 1024,  512, kernel_size=3) # 16
        self.up5 = PoolStackDecoder( 512,  512,  256, kernel_size=3) # 32
        self.up4 = PoolStackDecoder( 256,  256,  128, kernel_size=3) # 64
        self.up3 = PoolStackDecoder( 128,  128,   64, kernel_size=3) #128
        self.up2 = PoolStackDecoder(  64,   64,   32, kernel_size=3) #256
        self.up1 = PoolStackDecoder(  32,   32,   32, kernel_size=3) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self,x):

        #1024
        out = x

        #down0,out = self.down0(out) #512
        down1, out, i1 = self.down1(out)  #256
        down2, out, i2 = self.down2(out)  #128
        down3, out, i3 = self.down3(out)  #64
        down4, out, i4 = self.down4(out)  #32
        down5, out, i5 = self.down5(out)  #16
        down6, out, i6 = self.down6(out)  #8

        out = self.center(out)

        #debug
        if 0:
            print('x    ',x.size())
            print('down1',down1.size())
            print('down2',down2.size())
            print('down3',down3.size())
            print('down4',down4.size())
            print('down5',down5.size())
            print('down6',down6.size())
            print('out  ',out.size())     #, exit(0)

        out = self.up6(down6, out, i6)
        out = self.up5(down5, out, i5)
        out = self.up4(down4, out, i4)
        out = self.up3(down3, out, i3)
        out = self.up2(down2, out, i2)
        out = self.up1(down1, out, i1)
        #out = self.up0(down0, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 1
    C,H,W = 3,512,512  #640//2,960//2  #3,CARVANA_HEIGHT,CARVANA_WIDTH

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = MoreUNet(in_shape=(C,H,W)).cuda().train()
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


