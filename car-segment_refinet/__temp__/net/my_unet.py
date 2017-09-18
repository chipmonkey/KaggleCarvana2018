# unet from scratch
from common import *
from net.segmentation.loss   import *
from net.segmentation.blocks import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)

class ConvBnCRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True):
        super(ConvBnCRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels//2, eps=BN_EPS)
        self.relu = CReLU()
        if is_bn   is False: self.bn  =None
        if is_relu is False: self.relu=None


    def forward(self,x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x


    def merge_bn(self):
        if self.bn == None: return

        assert(self.conv.bias==None)
        conv_weight     = self.conv.weight.data
        bn_weight       = self.bn.weight.data
        bn_bias         = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var  = self.bn.running_var
        bn_eps          = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N,C,KH,KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat




## -----------------------------------------------------------------------------------------------------------

## origainl 3x3 stack filters used in UNet
class CStackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(CStackEncoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnCRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnCRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class CStackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(CStackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnCRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnCRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnCRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y
##---------------------------------------------------------------







## origainl 3x3 stack filters used in UNet
class ResStackEncoder1 (nn.Module):
    def __init__(self, x_channels, y_channels):
        super(ResStackEncoder1, self).__init__()
        self.encode1 = ConvBnRelu2d(x_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1)
        self.encode2 = ConvBnRelu2d(y_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1)

    def forward(self,x):
        y = self.encode1(x)
        y = self.encode2(y) + y
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class ResStackDecoder1 (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels):
        super(ResStackDecoder1, self).__init__()

        self.decode1 = ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=2, padding=1, dilation=1, stride=1, groups=1)
        self.decode2 = ConvBnRelu2d(y_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1)
        self.decode3 = ConvBnRelu2d(y_channels, y_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1)

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode1(y)
        y = self.decode2(y)+y
        y = self.decode3(y)+y
        return  y


# 512x512
class RefineUNet1024 (nn.Module):
    def __init__(self, in_shape):
        super(RefineUNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #512
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512,  768, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder( 768, 768, 512, kernel_size=3) #16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3) #32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3) #64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3) #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3) #256
        #512
        self.classify1 = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


        #--------------------------------------------------------------------------------
        self.mode='residual'  #experiments for different configuration
        #--------------------------------------------------------------------------------
        if self.mode=='simple1' or 'simple1_add':
            self.refine1 =  ConvBnRelu2d( 4,16)
            self.refine2 =  ConvBnRelu2d(16,16)
            self.refine3 =  ConvBnRelu2d(16,16)
            self.refine4 =  ConvBnRelu2d(16,16)
            D=16

        if self.mode=='residual':
            self.refine1 =  ConvBnRelu2d( 4,24)
            self.refine2 =  ConvResidual(24,24)
            self.refine3 =  ConvResidual(24,24)
            #self.refine4 =  ConvResidual(16,16)
            D=24

        if self.mode=='atrous':
            #use dilated conv
            self.refine1 =  ConvBnRelu2d( 4,20,dilation=1)
            self.refine2 =  ConvBnRelu2d(20,20,dilation=2,padding=2)
            self.refine3 =  ConvBnRelu2d(20,20,dilation=3,padding=3)
            D=20


        if self.mode=='densenet':
            #use denset concat style
            pass
        #--------------------------------------------------------------------------------
        self.classify2 = nn.Conv2d(D, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):
        N,C,H,W = x.size()
        #1024

        x_small = F.upsample(x,size=(512,512), mode='bilinear')
        down2,out = self.down2(x_small)   #;print('down2',down2.size())  #256
        down3,out = self.down3(out)       #;print('down3',down3.size())  #128
        down4,out = self.down4(out)       #;print('down4',down4.size())  #64
        down5,out = self.down5(out)       #;print('down5',down5.size())  #32
        down6,out = self.down6(out)       #;print('down6',down6.size())  #16
        pass                              #;print('out  ',out.size())

        out = self.center(out)       #16
        out = self.up6(down6, out)   #32
        out = self.up5(down5, out)   #64
        out = self.up4(down4, out)   #128
        out = self.up3(down3, out)   #256
        out = self.up2(down2, out)   #512
        out = self.classify1(out)
        y   = F.upsample(out,size=(H,W), mode='bilinear')

        #refine
        out = torch.cat((F.sigmoid(y),x),dim=1)

        #--------------------------------------------------------------------------------
        if self.mode=='simple1':
            out = self.refine1(out)
            out = self.refine2(out)
            out = self.refine3(out)
            out = self.refine4(out)

        if self.mode=='simple1_add':
            out = self.refine1(out)
            out = self.refine2(out) + out
            out = self.refine3(out) + out
            out = self.refine4(out) + out

        if self.mode=='residual':
            out = self.refine1(out)
            out = self.refine2(out)
            out = self.refine3(out)
            #out = self.refine4(out)

        if self.mode=='atrous':
            out = self.refine1(out)
            out = self.refine2(out) #;print(out.size())
            out = self.refine3(out) #;print(out.size())


        #--------------------------------------------------------------------------------
        out = self.classify2(out)
        out = out+y
        out = torch.squeeze(out, dim=1)
        return out



# 512x512
class DropUNet512 (nn.Module):
    def __init__(self, in_shape):
        super(DropUNet512, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = F.dropout(out,p=0.5,training=self.training)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out



# 512x512
class ResNet512 (nn.Module):
    def __init__(self, in_shape):
        super(ResNet512, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = ResStackEncoder(  C,   64 )   #256
        self.down3 = ResStackEncoder( 64,  128 )   #128
        self.down4 = ResStackEncoder(128,  256 )   #64
        self.down5 = ResStackEncoder(256,  512 )   #32
        self.down6 = ResStackEncoder(512, 1024 )   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = ResStackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = ResStackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = ResStackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = ResStackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = ResStackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        #out = F.dropout(out,p=0.5,training=self.training)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out




# 512x512
class ResNet512_1 (nn.Module):
    def __init__(self, in_shape):
        super(ResNet512_1, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = ResStackEncoder1(  C,   64 )   #256
        self.down3 = ResStackEncoder1( 64,  128 )   #128
        self.down4 = ResStackEncoder1(128,  256 )   #64
        self.down5 = ResStackEncoder1(256,  512 )   #32
        self.down6 = ResStackEncoder1(512, 1024 )   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = ResStackDecoder1(1024,1024, 512 )  # 16
        self.up5 = ResStackDecoder1( 512, 512, 256 )  # 32
        self.up4 = ResStackDecoder1( 256, 256, 128 )  # 64
        self.up3 = ResStackDecoder1( 128, 128,  64 )  #128
        self.up2 = ResStackDecoder1(  64,  64,  32 )  #256
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        #out = F.dropout(out,p=0.5,training=self.training)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out




# 512x512
class CUNet512 (nn.Module):
    def __init__(self, in_shape):
        super(CUNet512, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = CStackEncoder(  C,  128, kernel_size=3)   #256
        self.down3 = CStackEncoder(128,  256, kernel_size=3)   #128
        self.down4 = CStackEncoder(256,  512, kernel_size=3)   #64
        self.down5 = CStackEncoder(512,  512, kernel_size=3)   #32
        self.down6 = CStackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = CStackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = CStackDecoder( 512, 512, 512, kernel_size=3)  # 32
        self.up4 = CStackDecoder( 512, 512, 256, kernel_size=3)  # 64
        self.up3 = CStackDecoder( 256, 256, 128, kernel_size=3)  #128
        self.up2 = CStackDecoder( 128, 128,  64, kernel_size=3)  #256
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

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
        priors = labels

        net = CUNet512(in_shape=(C,H,W)).cuda().train()
        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        priors = Variable(priors).cuda()

        logits = net.forward(x)
        #logits = net.forward(x,priors)

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        print(net)
        print('logits')
        print(logits)
    #input('Press ENTER to continue.')


