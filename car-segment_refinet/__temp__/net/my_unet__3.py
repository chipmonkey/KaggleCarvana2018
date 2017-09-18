# unet from scratch
from common import *
from net.segmentation.loss   import *
from net.segmentation.blocks import *

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_norm_channel(x):
    z = torch.unsqueeze(torch.sum(x,dim=1),1)
    z2_mean = F.avg_pool2d(z*z,kernel_size=7,padding=3,stride=1)
    z_mean  = F.avg_pool2d(z,  kernel_size=7,padding=3,stride=1)
    z_std   = torch.sqrt(z2_mean-z_mean*z_mean+0.03)
    z = (z-z_mean)/(z_std)
    return z


def make_prior_channel(labels):
    a   = F.avg_pool2d(labels,kernel_size=11,padding=5,stride=1)
    return a


def make_ave_channel(x):
    a   = F.avg_pool2d(x,kernel_size=11,padding=5,stride=1)
    return a



# 128x128
class PriorUNet128 (nn.Module):
    def __init__(self, in_shape):
        super(PriorUNet128, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        C=3

        #128
        self.down3 = StackEncoder( C,   128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x, prior):
        out = x

        #add experimental channels -------------------------------------
        #z0 = make_norm_channel(x)
        #out = torch.cat((out,z0),1)

        # z0 = make_ave_channel(x)
        # out = torch.cat((out,z0),1)
        # labels = torch.unsqueeze(labels,1)
        # z0 = labels #make_ave_channel(labels)
        # out = torch.cat((out,z0),1)
        #add experimental channels -------------------------------------

        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.classify(out)+torch.unsqueeze(prior, dim=1)
        out = torch.squeeze(out, dim=1)
        return out


# 128x128
class LargeUNet128 (nn.Module):
    def __init__(self, in_shape):
        super(LargeUNet128, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #128
        self.down3 = StackEncoder( C,    256, kernel_size=3)   # 64
        self.down4 = StackEncoder( 256,  512, kernel_size=3)   # 32
        self.down5 = StackEncoder( 512, 1024, kernel_size=3)   # 16
        self.down6 = StackEncoder(1024, 2048, kernel_size=3)   #  8

        self.center = nn.Sequential(
            ConvBnRelu2d(2048, 2048, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(2048,2048,1024, kernel_size=3)  # 16
        self.up5 = StackDecoder(1024,1024, 512, kernel_size=3)  # 32
        self.up4 = StackDecoder( 512, 512, 256, kernel_size=3)  # 64
        self.up3 = StackDecoder( 256, 256, 128, kernel_size=3)  #128
        self.classify = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out


# 512x512
class LessUNet512 (nn.Module):
    def __init__(self, in_shape):
        super(LessUNet512, self).__init__()
        C,H,W = in_shape
        #assert(C==3)


        #1024
        #self.down2 = StackEncoder(  C,   64, kernel_size=3)   #128
        self.down3 = StackEncoder(  C,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        #self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        #down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        #out = self.up2(down2, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out






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
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3) #16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3) #32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3) #64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3) #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3) #256
        #512
        self.classify1 = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)


        #--------------------------------------------------------------------------------
        self.mode='simple1_add'  #experiments for different configuration
        #--------------------------------------------------------------------------------
        if self.mode=='simple1' or 'simple1_add':
            self.refine1 =  ConvBnRelu2d( 4,16)
            self.refine2 =  ConvBnRelu2d(16,16)
            self.refine3 =  ConvBnRelu2d(16,16)
            self.refine4 =  ConvBnRelu2d(16,16)
            D=16

        if self.mode=='residual':
            self.refine1 =  ConvBnRelu2d( 4,16)
            self.refine2 =  ConvResidual(16,16)
            self.refine3 =  ConvResidual(16,16)
            self.refine4 =  ConvResidual(16,16)
            D=16

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
class RefineUNet1024_0 (nn.Module):
    def __init__(self, in_shape):
        super(RefineUNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #512
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )


        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3) #16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3) #32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3) #64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3) #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3) #256
        #512

        self.classify1 = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.refine= nn.Sequential(
            ConvBnRelu2d(4, 24),
            ConvBnRelu2d(24,24),
        )
        self.classify2 = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True)


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
        out = torch.cat((y,x),dim=1)
        out = self.refine(out)
        out = self.classify2(out)
        out = out+y

        out = torch.squeeze(out, dim=1)
        return out

# 1024x1024
class UNet1024 (nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)


        #1024
        self.down0 = StackEncoder(  C,   16, kernel_size=3)   #512
        self.down1 = StackEncoder( 16,   32, kernel_size=3)   #256
        self.down2 = StackEncoder( 32,   64, kernel_size=3)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8


        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.up1 = StackDecoder(  32,  32,  16, kernel_size=3)  #512
        self.up0 = StackDecoder(  16,  16,  16) #1024

        self.classify = nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down0,out = self.down0(out)  ##;print('down0',down0.size())  #512
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        out = self.up0(down0, out)
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
    C,H,W = 3,1024,1024  #640//2,960//2  #3,CARVANA_HEIGHT,CARVANA_WIDTH

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)
        priors = labels

        net = RefineUNet1024(in_shape=(C,H,W)).cuda().train()
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


