# unet from scratch
from common import *
from net.segmentation.loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# baseline 128x128, 256x256, 512x512 for experiments -----------------------------------------------

BN_EPS = 1e-4  #1e-4  #1e-5



def merge_bn_in_net(net):
    print ('merging bn ....')
    for m in net.modules():
        if isinstance(m, (StackEncoder, StackDecoder)):
            for mm in m.modules():
                if isinstance(mm, (ConvBnRelu2d,)):
                    print('merging ...')
                    mm.merge_bn()



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
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y
##---------------------------------------------------------------


## origainl 3x3 stack filters used in UNet
class ResStackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels):
        super(ResStackEncoder, self).__init__()
        self.encode = ConvResidual(x_channels, y_channels)

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class ResStackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(ResStackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvResidual(y_channels, y_channels)
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y


##---------------------------------------------------------------

def make_check_cbr():

    C,H,W = 1,5,5
    input = np.zeros((1,C,H,W), np.float32)
    input[0,0,2,2]=1
    #input[0,0,4,5]=1
    conv_w = np.array([[[[1,2,3],[0,0,0],[0,0,0]]]], np.float32)
    bn_w    = np.array([1], np.float32)
    bn_b    = np.array([0], np.float32)
    bn_mean = np.array([0], np.float32)
    bn_var  = np.array([1], np.float32)

    if 0:
        for x in range(W):
            for y in range(H):
                if x%4==0 and y%4==0:
                    input[0,y,x]=1

    cbr = ConvBnRelu2d(C, 1).eval()
    cbr.conv.weight.data     = torch.from_numpy(conv_w)
    cbr.bn.weight.data       = torch.from_numpy(bn_w)
    cbr.bn.bias.data         = torch.from_numpy(bn_b)
    #cbr.bn.running_mean.data = torch.from_numpy(bn_mean)
    #cbr.bn.running_var.data  = torch.from_numpy(bn_var)

    cbr.bn.eps=15

    assert(cbr.conv.bias==None)

    #cbr.bn
    '''
    cbr.bn.state_dict()
    OrderedDict([('weight', 
     0.9219
    [torch.FloatTensor of size 1]
    ), ('bias', 
     0
    [torch.FloatTensor of size 1]
    ), ('running_mean', 
     0
    [torch.FloatTensor of size 1]
    ), ('running_var', 
     1
    [torch.FloatTensor of size 1]
    )])
    '''

    input = Variable(torch.from_numpy(input))
    output = cbr(input)
    output = output.data


    #conv.weight.data = torch.from_numpy(w)
    ## initialistaion
    #  https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    #  bang liu : random normal(mean=0,std=filter_width*filter_height*channel).

    # for m in self.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))

    # input = Variable(torch.from_numpy(input))
    # output = conv(input)
    # output = output.data
    print('input',input)
    print('output',output)
    print('cbr.conv.weight',cbr.conv.weight,'\n')
    print('cbr.bn.weight',cbr.bn.weight,'\n')
    print('cbr.bn.bias',  cbr.bn.bias,'\n')
    print('cbr.bn.running_mean',cbr.bn.running_mean,'\n')
    print('cbr.bn.running_var', cbr.bn.running_var,'\n')
    print('cbr.bn.eps', cbr.bn.eps,'\n')





def make_dummy_cbr():
    N,C,H,W = 10,3,5,5
    D=2

    #input = np.zeros((N,C,H,W), np.float32)
    inputs  = np.random.uniform( low=-1, high=1, size=(N,C,H,W)).astype(np.float32)
    targets = np.random.uniform( low=-1, high=1, size=(N,D,H,W)).astype(np.float32)

    cbr = ConvBnRelu2d(C, D)
    optimizer = optim.SGD(cbr.parameters(), lr=0.01, momentum=0.9, weight_decay=0)

    cbr.train()
    for it in range(3):
        x     = Variable(torch.from_numpy(inputs))
        y_hat = Variable(torch.from_numpy(targets))
        y = cbr(x)

        loss = nn.MSELoss()(y,y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data[0]
        print('%02d: %5.5f'%(it,batch_loss))


    #save
    cbr.eval()
    out_dir='/root/share/project/kaggle-carvana-cars/results/yyy'
    os.makedirs(out_dir+'/train', exist_ok=True)


    torch.save(cbr.state_dict(), out_dir +'/train/cbr.state_dict')
    x     = Variable(torch.from_numpy(inputs))
    y_hat = Variable(torch.from_numpy(targets))
    y     = cbr(x)
    loss = nn.MSELoss()(y,y_hat)
    print(loss)

    np.savetxt(out_dir +'/train/inputs.txt', x.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/train/targets.txt',y_hat.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/train/outputs.txt',y.data.numpy().reshape(-1),fmt='%0.5f')

    np.save(out_dir +'/train/inputs.npy', x.data.numpy())
    np.save(out_dir +'/train/targets.npy',y_hat.data.numpy())
    np.save(out_dir +'/train/outputs.npy',y.data.numpy())


# def check_dummy_cbr():
#     N,C,H,W = 10,1,5,5
#     D=1
#
#     #input = np.zeros((N,C,H,W), np.float32)
#     out_dir='/root/share/project/kaggle-carvana-cars/results/xxx'
#     inputs   = np.load(out_dir +'/train/inputs.npy')
#     targets  = np.load(out_dir +'/train/targets.npy')
#     outputs0 = np.load(out_dir +'/train/outputs.npy')
#
#     cbr = ConvBnRelu2d(C, D)
#     cbr.load_state_dict(torch.load(out_dir +'/train/cbr.state_dict'))
#     cbr.eval()
#
#     #save
#     out_dir='/root/share/project/kaggle-carvana-cars/results/xxx/check'
#     torch.save(cbr.state_dict(), out_dir +'/cbr.state_dict')
#     x     = Variable(torch.from_numpy(inputs))
#     y_hat = Variable(torch.from_numpy(targets))
#     y     = cbr(x)
#     loss = nn.MSELoss()(y,y_hat)
#     print(loss)
#
#     np.savetxt(out_dir +'/inputs.txt', x.data.numpy().reshape(-1),fmt='%0.5f')
#     np.savetxt(out_dir +'/targets.txt',y_hat.data.numpy().reshape(-1),fmt='%0.5f')
#     np.savetxt(out_dir +'/outputs.txt',y.data.numpy().reshape(-1),fmt='%0.5f')
#


def check_dummy_cbr1():
    N,C,H,W = 10,3,5,5
    D=2

    #input = np.zeros((N,C,H,W), np.float32)
    out_dir='/root/share/project/kaggle-carvana-cars/results/yyy'
    inputs   = np.load(out_dir +'/train/inputs.npy')
    targets  = np.load(out_dir +'/train/targets.npy')
    outputs0 = np.load(out_dir +'/train/outputs.npy')

    cbr = ConvBnRelu2d(C, D,is_relu = False)
    cbr.load_state_dict(torch.load(out_dir +'/train/cbr.state_dict'))

    conv_weight = cbr.conv.weight.data
    assert(cbr.conv.bias==None)
    bn_weight = cbr.bn.weight.data
    bn_bias   = cbr.bn.bias.data
    bn_running_mean = cbr.bn.running_mean
    bn_running_var  = cbr.bn.running_var
    bn_eps          = cbr.bn.eps

    print(conv_weight)
    print(bn_weight)
    print(bn_bias)
    print(bn_running_mean)
    print(bn_running_var)
    print(bn_eps)

    #https://github.com/sanghoon/pva-faster-rcnn/issues/5
    #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c
    N,C,KH,KW = conv_weight.size()
    #conv_weight_sum = conv_weight.view(N,-1).sum(dim=1)
    std = 1/(torch.sqrt(bn_running_var+bn_eps))

    std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
    conv_weight_hat = std_bn_weight*conv_weight
    conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)#*conv_weight_sum

    conv_hat = nn.Conv2d(in_channels=cbr.conv.in_channels, out_channels=cbr.conv.out_channels, kernel_size=cbr.conv.kernel_size, padding=cbr.conv.padding, stride=cbr.conv.stride, dilation=cbr.conv.dilation, groups=cbr.conv.groups, bias=True)
    #fill in
    conv_hat.weight.data = conv_weight_hat
    conv_hat.bias.data   = conv_bias_hat

    #save-------------------------------------------------------------------------
    cbr.eval()
    os.makedirs(out_dir+'/check', exist_ok=True)


    torch.save(cbr.state_dict(), out_dir +'/check/cbr.state_dict')
    x     = Variable(torch.from_numpy(inputs))
    y_hat = Variable(torch.from_numpy(targets))
    y     = cbr(x)
    loss = nn.MSELoss()(y,y_hat)
    print('\nloss\n',loss)


    ##
    y1=conv_hat(x)
    #print('\ny1\n',y1)

    np.savetxt(out_dir +'/check/inputs.txt', x.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/check/targets.txt',y_hat.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/check/outputs.txt',y.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/check/outputs1.txt',y1.data.numpy().reshape(-1),fmt='%0.5f')




def check_dummy_cbr2():
    N,C,H,W = 10,3,5,5
    D=2

    #input = np.zeros((N,C,H,W), np.float32)
    out_dir='/root/share/project/kaggle-carvana-cars/results/yyy'
    inputs   = np.load(out_dir +'/train/inputs.npy')
    targets  = np.load(out_dir +'/train/targets.npy')
    outputs0 = np.load(out_dir +'/train/outputs.npy')

    cbr = ConvBnRelu2d(C, D,is_relu = False)
    cbr.load_state_dict(torch.load(out_dir +'/train/cbr.state_dict'))


    #save-------------------------------------------------------------------------
    cbr.eval()
    os.makedirs(out_dir+'/check2', exist_ok=True)

    torch.save(cbr.state_dict(), out_dir +'/check2/cbr.state_dict')
    x     = Variable(torch.from_numpy(inputs))
    y_hat = Variable(torch.from_numpy(targets))
    y     = cbr(x)
    loss = nn.MSELoss()(y,y_hat)
    print('\nloss\n',loss)


    ##
    cbr.merge_bn()
    y1=cbr(x)

    np.savetxt(out_dir +'/check2/inputs.txt', x.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/check2/targets.txt',y_hat.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/check2/outputs.txt',y.data.numpy().reshape(-1),fmt='%0.5f')
    np.savetxt(out_dir +'/check2/outputs1.txt',y1.data.numpy().reshape(-1),fmt='%0.5f')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #make_dummy_cbr()
    #make_dummy_cbr()
    check_dummy_cbr2()

    print('\nsucess!')