# unet from scratch

from common import *
from net.segmentation.loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# http://pytorch.org/docs/master/nn.html#torch.nn.utils.weight_norm
## see rtqichen/example.py ( python)
#import torch.nn.utils.weight_norm as weight_norm
#def weight_norm(x): return x

BN_EPS = 1e-4  #1e-4  #1e-5

# https://github.com/zsdonghao/tensorlayer/issues/53
# How to initialize DeConv2dLayer weights to bilinear upsampling?
def bilinear_filler(filter_shape, upscale_factor):
        #filter_shape = out_c, in_c, in_kh, in_kw
        out_c, kc, kh, kw =  filter_shape
        sy,sx =  upscale_factor
        #assert(out_c==in_c)

        if kh % 2 == 1:
            y_centre = sy - 1
        else:
            y_centre = sy - 0.5

        if kw % 2 == 1:
            x_centre = sx - 1
        else:
            x_centre = sx - 0.5


        bilinear = np.zeros((kh, kw), np.float32)
        for x in range(kw):
            for y in range(kh):
                ##Interpolation Calculation
                value = (1 - abs((x - x_centre)/ sx)) * (1 - abs((y - y_centre)/ sy))
                bilinear[x, y] = value


        weights = np.zeros((kc, out_c, kh, kw), np.float32)
        if kc == out_c:
            for i in range(out_c):
                weights[i, i, :, :] = bilinear
        else:
            for i in range(kc):
                for j in range(out_c):
                    weights[i, j, :, :] = bilinear/kc

        return weights

#-------------------------------------------------------------------------------------------

def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels, eps=BN_EPS),  #eps=1e-5
        nn.ReLU(inplace=True),
    ]


## -----------------------------------------------------------------------------------------------------------

## origainl 3x3 stack filters used in UNet
class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels):
        super(StackEncoder, self).__init__()
        self.encode0 = nn.Sequential(
            nn.Conv2d(x_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels,eps=BN_EPS),
            nn.ReLU(inplace=True) #nn.ELU(inplace=True)  #   nn.PReLU(), #
        )
        self.encode1 = nn.Sequential(
            nn.Conv2d(y_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels,eps=BN_EPS),
            nn.ReLU(inplace=True) ##nn.ELU(inplace=True)  #   nn.PReLU(),
        )

    def forward(self,x):
        y = self.encode0(x)
        y = self.encode1(y)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels):
        super(StackDecoder, self).__init__()

        self.mix = nn.Sequential(
            nn.Conv2d(x_big_channels+x_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels,eps=BN_EPS),  #eps=1e-5
            nn.ReLU(inplace=True),
        )
        self.decode0 = nn.Sequential(
            nn.Conv2d(y_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels,eps=BN_EPS),
            nn.ReLU(inplace=True)  #   nn.PReLU(),  #nn.ReLU
        )
        self.decode1 = nn.Sequential(
            nn.Conv2d(y_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels,eps=BN_EPS),
            nn.ReLU(inplace=True)  #   nn.PReLU(),
        )

    def forward(self, x_big, x):
        y = torch.cat([
            x_big,
            F.upsample(x, scale_factor=2,mode='bilinear')
        ],1)
        y = self.mix(y)
        y = self.decode0(y)
        y = self.decode1(y)
        return  y
## -----------------------------------------------------------------------------------------------------------


## 'Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes'
class FRRU (nn.Module):
    def __init__(self, y_previous_channels, z_previous_channels, y_channels, z_channels):
        super(FRRU, self).__init__()
        assert(z_previous_channels==z_channels)

        self.decode0 = nn.Sequential(
            nn.Conv2d(z_previous_channels+y_previous_channels, y_channels, kernel_size=3, padding=1,  stride=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels),
            nn.ReLU(inplace=True)  #   nn.PReLU(),  #nn.ReLU
        )
        self.decode1 = nn.Sequential(
            nn.Conv2d(y_channels, y_channels, kernel_size=3, padding=1,  stride=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(y_channels),
            nn.ReLU(inplace=True)  #   nn.PReLU(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(y_channels, z_channels, kernel_size=1, padding=1,  stride=1, dilation=1, groups=1, bias=True),
        )

    def forward(self, y_previous, z_previous):

        y = torch.cat([y_previous,y],1)
        y = self.decode0(y)
        y = self.decode1(y)

        z = self.residual(y)
        z = F.upsample(z, size=self.size, scale_factor=self.scale_factor,mode='bilinear')
        z = z+z_previous
        return  y, z



## -----------------------------------------------------------------------------------------------------------
class StackWeightNormEncoder (nn.Module):
    def __init__(self, in_channels, out_channels, t=-1):
        super(StackWeightNormEncoder, self).__init__()
        self.t=t
        self.encode0 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True)),
        )
        self.encode1 = nn.Sequential(
            weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,  stride=1, dilation=1, groups=1, bias=True)),
        )

    def forward(self,x):
        t = self.t

        z = self.encode0(x)
        z = F.relu(z-t,inplace=True)+t
        z = self.encode1(z)
        z = F.relu(z-t,inplace=True)+t
        z_small = F.max_pool2d(z, kernel_size=2, stride=2)
        return z, z_small


class StackWeightNormDecoder (nn.Module):
    def __init__(self, in_large_channels, in_channels, out_channels, t=-1):
        super(StackWeightNormDecoder, self).__init__()
        self.t=t

        self.mix = nn.Sequential(
            nn.Conv2d(in_large_channels+in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            #nn.ReLU(inplace=True),
        )
        self.decode0 = nn.Sequential(
            weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True)),
        )
        self.decode1 = nn.Sequential(
            weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,  stride=1, dilation=1, groups=1, bias=True)),
        )

    def forward(self, x_large, x):
        t = self.t

        z = torch.cat([
            x_large,
            F.upsample(x, scale_factor=2,mode='bilinear')
        ],1)

        z = self.mix(z)
        z = self.decode0(z)
        z = F.relu(z-t,inplace=True)+t
        z = self.decode1(z)
        z = F.relu(z-t,inplace=True)+t
        return  z
## -----------------------------------------------------------------------------------------------------------






#resnext
class StackEncoderEx (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoderEx, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(128),  #eps=1e-5
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),  #eps=1e-5
        )
        self.shortcut = None
        if in_channels!=out_channels or stride!=1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,stride=1,  bias=True)

    def forward(self,x):

        z = self.encode(x)
        x = x if self.shortcut is None else self.shortcut(x)
        z = F.relu(z+x)

        z_small = F.max_pool2d(z, kernel_size=2, stride=2)
        return z, z_small


class StackDecoderEx (nn.Module):
    def __init__(self, in_large_channels, in_channels, out_channels):
        super(StackDecoderEx, self).__init__()

        self.mix = nn.Sequential(
            nn.Conv2d(in_large_channels+in_channels, in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),  #eps=1e-5
            nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(128),  #eps=1e-5
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),  #eps=1e-5
        )
        self.shortcut = None
        if in_channels!=out_channels or stride!=1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,stride=1,  bias=True)

    def forward(self, x_large, x):
        x = torch.cat([
            x_large,
            F.upsample_bilinear(x, scale_factor=2)
        ],1)
        x = self.mix(x)

        z = self.decode(x)
        x = x if self.shortcut is None else self.shortcut(x)
        z = F.relu(z+x)
        return z

# "Convolutional Neural Pyramid for Image Processing"
# - X Shen, YC Chen, X Tao, J Jia - arXiv preprint arXiv:1704.02071, 2017
class ShrinkExpandEncoder (nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ShrinkExpandEncoder, self).__init__()
        c = in_channels
        self.encode = nn.Sequential(
            *make_conv_bn_relu(in_channels, c,  kernel_size=1, padding=0, stride=1 ),
            *make_conv_bn_relu(c, c, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(c, out_channels, kernel_size=1, padding=0, stride=1 ),
        )
    def forward(self,x):

        z       = self.encode(x)
        z_small = F.max_pool2d(z, kernel_size=2, stride=2)
        return z, z_small


##-------------------------------------------------------------------------------------------------------------



class ResidualBlock (nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels, eps=BN_EPS),  #eps=1e-5
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1,  bias=False),
            nn.BatchNorm2d(out_channels, eps=BN_EPS),
        )
        self.shortcut = None
        if in_channels!=out_channels or stride!=1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride,  bias=True)

    def forward(self,x):
        out = self.block(x)
        x = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(out+x)
        return out


class ResidualBottleneckBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBottleneckBlock, self).__init__()

        self.block = nn.Sequential(
            *make_conv_bn_relu( in_channels, in_channels,  kernel_size=1, padding=0, stride=1 ),
            *make_conv_bn_relu( in_channels, in_channels,  kernel_size=3, padding=1, stride=1 ),
            nn.Conv2d         ( in_channels, out_channels, kernel_size=1, padding=1, stride=0, bias=False),
            nn.BatchNorm2d(out_channels),  #eps=1e-5
        )
        self.shortcut = None
        if(in_channels!=out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,stride=1,  bias=True)

    def forward(self,x):
        out = self.block(x)
        x = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(out+x)
        return out

## 'LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation' - Abhishek Chaurasia
class ResidualEncoder (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualEncoder, self).__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),  #eps=1e-5
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=True),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),  #eps=1e-5
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self,x):
        z = x
        z_small = F.relu(self.encode1(z)       + self.shortcut1(z), inplace=True)
        z_small = F.relu(self.encode2(z_small) + z_small,           inplace=True)

        return z, z_small


class ResidualDecoder (nn.Module):

    def __init__(self, in_large_channels, in_channels, out_channels):
        super(ResidualDecoder, self).__init__()
        assert (in_channels%4==0)

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(in_channels//4),  #eps=1e-5
            nn.ReLU(inplace=True),  #1, 128, 8, 8

            nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//4, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # for m in self.decode.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         w = bilinear_filler(filter_shape=(m.out_channels,m.in_channels,*m.kernel_size), upscale_factor = m.stride)
        #         m.weight.data = torch.from_numpy(w)

    def forward(self,x_large,x):
        out = self.decode(x)
        out = out + x_large if x_large is not None else out
        out = F.relu(out, inplace=True)
        return  out

##---------------------------------------------------------------------------------------


class StackSlimEncoder (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoder, self).__init__()
        self.encode = nn.Sequential(
            *make_conv_bn_relu( in_channels, out_channels, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(out_channels, out_channels, kernel_size=3, padding=1, stride=1 ),
        )

    def forward(self,x):
        z       = self.encode(x)
        z_small = F.max_pool2d(z, kernel_size=2, stride=2)
        return z, z_small


class StackSlimDecoder (nn.Module):
    def __init__(self, in_large_channels, in_channels, out_channels):
        super(StackDecoder, self).__init__()

        self.mix = nn.Sequential(
            nn.Conv2d(in_large_channels+in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            #nn.Conv2d(in_large_channels+in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            #nn.BatchNorm2d(out_channels),  #eps=1e-5
            #nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            *make_conv_bn_relu(out_channels,out_channels, kernel_size=3, padding=1, stride=1 ),
            *make_conv_bn_relu(out_channels,out_channels, kernel_size=3, padding=1, stride=1 ),
        )

    def forward(self, x_large, x):
        z = torch.cat([
            x_large,
            F.upsample_bilinear(x, scale_factor=2)
        ],1)
        z = self.mix(z)
        z = self.decode(z)
        return  z



##---------------------------------------------------------------------------------------

class RefineUNet512(nn.Module):
    def __init__(self, in_shape):
        super(RefineUNet512, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)

        #512
        self.down1 = StackEncoder(  3, 32)   #256
        self.down2 = StackEncoder( 32, 32)   #128
        self.down3 = StackEncoder( 32, 64)   # 64
        self.down4 = StackEncoder( 64,128)   # 32
        self.down5 = StackEncoder(128,256)   # 16
        self.down6 = StackEncoder(256,512)   #  8

        self.center0 = nn.Sequential(
            weight_norm(nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),## nn.ELU(inplace=True),  #nn.ReLU(inplace=True),
        )
        self.center1 = nn.Sequential(
            weight_norm(nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),## nn.ELU(inplace=True),  #nn.ReLU(inplace=True),
        )

        # 8
        self.up6 = StackDecoder(512,1024, 512) # 16
        self.up5 = StackDecoder(256, 512, 256) # 32
        self.up4 = StackDecoder(128, 256, 128) # 64
        self.up3 = StackDecoder(64,  128,  64) #128
        self.up2 = StackDecoder(32,   64,  32) #256
        self.up1 = StackDecoder(32,   32,  32) #512


        # for m in self.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         w = bilinear_filler(filter_shape=(m.out_channels,m.in_channels,*m.kernel_size), upscale_factor = m.stride)
        #         m.weight.data = torch.from_numpy(w)



        # self.refine0 = nn.Sequential(
        #      *make_conv_bn_relu(3, 16, kernel_size=5, padding=2, stride=1 ),
        # )
        # self.refine1 = nn.Sequential(
        #     *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        #     *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        # )
        # self.refine2 = nn.Sequential(
        #     *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        #     *make_conv_bn_relu(32, 32, kernel_size=3, padding=1, stride=1 ),
        # )
        self.classify1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)
        )

        # self.classify1 = nn.Sequential(
        #     *make_conv_relu(32, 1, kernel_size=5, padding=2, stride=1 ),
        # )
        # self.classify2 = nn.Sequential(
        #     *make_conv_relu(1, 1, kernel_size=5, padding=2, stride=1 ),
        # )
        #self.classify1 = nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1 )


    def forward(self, x):

        sample = F.upsample


        #512
        out = sample(x,size=(512, 512),mode='bilinear')
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center0(out)
        out = self.center1(out)

        # t=-0.1
        # out = F.relu(self.center0(out)-t,inplace=True)+t
        # out = F.relu(self.center1(out)-t,inplace=True)+t
        ## out = self.center(out)

        #debug
        # print(x.size())
        # print(down1.size())
        # print(down2.size())
        # print(down3.size())
        # print(down4.size())
        # print(down5.size())
        # print(down6.size())
        # print(out.size())     #, exit(0)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #512

        # refinement
        # out = upsample_bilinear  (out,size=(self.H//2, self.W//2))
        # x   = downsample_bilinear(x,  size=(self.H//2, self.W//2))
        # x   = self.refine0(x)
        # out = torch.cat([x, out],1)
        #
        # out = self.refine1(out)
        # out = self.refine2(out)

        out = self.classify1(out)
        out = sample(out,size=(self.H, self.W),mode='bilinear')
        out = torch.squeeze(out, dim=1)
        return out



class FRRUNet512(nn.Module):
    def __init__(self, in_shape):
        super(FRUNet512, self).__init__()
        self.C = in_shape[0]
        self.H = in_shape[1]
        self.W = in_shape[2]
        assert(self.C==3)


        #512
        self.preprocess = nn.Sequential(
            *make_conv_bn_rel(512,32,kernel_size=5,padding=2, stride=1),
            ResidualBlock(32,32),
            ResidualBlock(32,32),
        )

        self.down1 = FRRU( 32, 32,  32, 32)   #256
        self.down2 = FRRU( 32, 32,  32, 32)   #128
        self.down3 = FRRU( 32, 32,  64, 32)   # 64
        self.down4 = FRRU( 64, 32, 128, 32)   # 32
        self.down5 = FRRU(128, 32, 256, 32)   # 16
        self.down6 = FRRU(256, 32, 512, 32)   #  8

        # 8
        self.up5 = FRRU(256, 32, 512, 32) # 32
        # self.up4 = FRRU(128, 256, 128) # 64
        # self.up3 = FRRU(64,  128,  64) #128
        # self.up2 = FRRU(32,   64,  32) #256
        # self.up1 = FRRU(32,   32,  32) #512

        self.lassify1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)
        )

        # self.classify1 = nn.Sequential(
        #     *make_conv_relu(32, 1, kernel_size=5, padding=2, stride=1 ),
        # )
        # self.classify2 = nn.Sequential(
        #     *make_conv_relu(1, 1, kernel_size=5, padding=2, stride=1 ),
        # )
        #self.classify1 = nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1 )


    def forward(self, x):

        #512
        z = F.upsample(x,size=(512, 512),mode='bilinear')
        y = z

        #down ---
        y, i1 = F.max_pool2d(y,kernel_size=2,stride=2, return_indices=True) #256
        z, y  = self.down1(z, y) ## FRRU

        y, i2 = F.max_pool2d(y,kernel_size=2,stride=2, return_indices=True) #128
        z, y  = self.down2(z, y)

        y, i3 = F.max_pool2d(y,kernel_size=2,stride=2, return_indices=True) # 64
        z, y  = self.down3(z, y)

        y, i4 = F.max_pool2d(y,kernel_size=2,stride=2, return_indices=True) # 32
        z, y  = self.down4(z, y)

        y, i5 = F.max_pool2d(y,kernel_size=2,stride=2, return_indices=True) # 16
        z, y  = self.down5(z, y)

        y, i6 = F.max_pool2d(y,kernel_size=2,stride=2, return_indices=True) #  8
        z, y  = self.down6(z, y)


        # up ---
        y = F.max_unpool2d(z,i6,kernel_size=2,stride=2) #  8
        z, y = self.up6(z, y)

        y = F.max_unpool2d(z,i5,kernel_size=2,stride=2) #  8
        z, y = self.up5(z, y)

        y = F.max_unpool2d(z,i4,kernel_size=2,stride=2) #  8
        z, y = self.up4(z, y)

        y = F.max_unpool2d(z,i3,kernel_size=2,stride=2) #  8
        z, y = self.up3(z, y)

        y = F.max_unpool2d(z,i2,kernel_size=2,stride=2) #  8
        z, y = self.up2(z, y)

        y = F.max_unpool2d(z,i1,kernel_size=2,stride=2) #  8
        z, y = self.up1(z, y)




        #512

        # refinement
        # out = upsample_bilinear  (out,size=(self.H//2, self.W//2))
        # x   = downsample_bilinear(x,  size=(self.H//2, self.W//2))
        # x   = self.refine0(x)
        # out = torch.cat([x, out],1)
        #
        # out = self.refine1(out)
        # out = self.refine2(out)

        out = self.classify1(out)
        out = sample(out,size=(self.H, self.W),mode='bilinear')
        out = torch.squeeze(out, dim=1)
        return out











##---------------------------------------------------------------------------------------

class UNet1024(nn.Module):
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

        self.center0 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024,eps=BN_EPS),
            nn.ReLU(inplace=True),## nn.ELU(inplace=True),  #nn.ReLU(inplace=True),
        )
        self.center1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024,eps=BN_EPS),
            nn.ReLU(inplace=True),
        )

        # 8
        self.up6 = StackDecoder(512,1024, 512) # 16
        self.up5 = StackDecoder(256, 512, 256) # 32
        self.up4 = StackDecoder(128, 256, 128) # 64
        self.up3 = StackDecoder(64,  128,  64) #128
        self.up2 = StackDecoder(32,   64,  32) #256
        self.up1 = StackDecoder(16,   32,  16) #512
        #self.up0 = StackDecoder(16,   16,  16) #1024

        self.classify1 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1, bias=True)
        )

    def forward(self, x):

        #1024
        out = F.upsample(x,size=(512, 512),mode='bilinear')
        if 1: #add norm channel
            z = out
            z = torch.unsqueeze(torch.sum(z,dim=1),1)
            z2_mean = F.avg_pool2d(z*z,kernel_size=7,padding=3,stride=1)
            z_mean  = F.avg_pool2d(z,  kernel_size=7,padding=3,stride=1)
            z_std   = torch.sqrt(z2_mean-z_mean*z_mean+0.03)
            z = (z-z_mean)/(z_std)
            out = torch.cat([z,out],1)
            z = None

        #down0,out = self.down0(out)  #512
        down1,out = self.down1(out)  #256
        down2,out = self.down2(out)  #128
        down3,out = self.down3(out)  #64
        down4,out = self.down4(out)  #32
        down5,out = self.down5(out)  #16
        down6,out = self.down6(out)  #8

        out = self.center0(out)
        out = self.center1(out)

        #debug
        # print(x.size())
        # print(down1.size())
        # print(down2.size())
        # print(down3.size())
        # print(down4.size())
        # print(down5.size())
        # print(down6.size())
        # print(out.size())     #, exit(0)

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





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 4
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


        #BN 6485 : iter 350 0.9619
        #BN 6389 : 2350 0.9619