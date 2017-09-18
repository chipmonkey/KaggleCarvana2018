# https://github.com/facebookresearch/ResNeXt/blob/master/README.md
# python convert_torch.py -m vgg16.t7
#  https://github.com/cypw/ResNeXt-1

from net.common import *
from functools import reduce

#--------------------------------------------------------------------------------
# https://github.com/amdegroot/pytorch-containers
# https://discuss.pytorch.org/t/are-tables-like-concattable-or-paralleltable-present-in-torch/128/2



# ---conv-bn-relu----conv(group)-bn-relu-----conv-bn--(+)-- relu
#  |                                                   |
#  |-----------(identity or conv-b )-------------------|

class Block(nn.Module):
    def __init__(self, in_channels, channels=[128,128,256], stride=1,shortcut=True, groups=32):
        super(Block, self).__init__()
        self.conv=nn.Sequential( # Sequential,
            nn.Conv2d(in_channels,channels[0],kernel_size=1,stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1],channels[2],kernel_size=1,stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(channels[2]),
        )
        if shortcut:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,channels[2],kernel_size=1,stride=stride, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(channels[2]),
            )
        else:
            self.shortcut=None

    def forward(self,x):
        y = self.conv(x)
        if self.shortcut is not None:
            x = self.shortcut(x)

        return x,y



class ResNext50(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(ResNext50, self).__init__()
        in_channels, height, width = in_shape

        ##resnext_50_32x4d
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
        )

        self.blocks = nn.ModuleList()
        self.blocks.add_module('conv2a', Block( 64,[128,128,256],1,True, 32))
        self.blocks.add_module('conv2b', Block(256,[128,128,256],1,False,32))
        self.blocks.add_module('conv2c', Block(256,[128,128,256],1,False,32))#2

        self.blocks.add_module('conv3a', Block(256,[256,256,512],2,True, 32))
        self.blocks.add_module('conv3b', Block(512,[256,256,512],1,False,32))
        self.blocks.add_module('conv3c', Block(512,[256,256,512],1,False,32))
        self.blocks.add_module('conv3d', Block(512,[256,256,512],1,False,32))#6

        self.blocks.add_module('conv4a', Block( 512,[512,512,1024],2,True, 32))
        self.blocks.add_module('conv4b', Block(1024,[512,512,1024],1,False,32))
        self.blocks.add_module('conv4c', Block(1024,[512,512,1024],1,False,32))
        self.blocks.add_module('conv4e', Block(1024,[512,512,1024],1,False,32))
        self.blocks.add_module('conv4f', Block(1024,[512,512,1024],1,False,32))
        self.blocks.add_module('conv4g', Block(1024,[512,512,1024],1,False,32))#12

        self.blocks.add_module('conv5a', Block(1024,[1024,1024,2048],2,True, 32))
        self.blocks.add_module('conv5b', Block(2048,[1024,1024,2048],1,False,32))
        self.blocks.add_module('conv5c', Block(2048,[1024,1024,2048],1,False,32))#15

        # ---------------------------------
        num_groups = 412

        # config-a
        self.fc1 = nn.Linear(2048, num_groups)
        self.fc  = nn.Linear(2048, num_classes)


    #  https://arxiv.org/abs/1603.09382
    #  Deep Networks with Stochastic Depth
    def forward(self, x):

        # num_blocks=len(self.blocks)
        # survials = np.zeros(num_blocks,np.float32)
        # if self.training==True:
        #     for n in range(num_blocks):
        #         survials[n] =  n/num_blocks
        # x  = self.conv(x)
        # for n,block in enumerate(self.blocks):
        #     x,y = block(x)
        #     a = float(np.random.uniform()>survials[n])
        #     x = x + a*y
        #     x = F.relu(x,inplace=True)

        x  = self.conv(x)
        for n,block in enumerate(self.blocks):
            x,y = block(x)
            x = x + y
            x = F.relu(x,inplace=True)
            #if n in [6,12,15]:
            #    x = F.dropout(x,p=0.10,training=self.training)

        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = x.view(x.size(0), -1)
        #x = F.dropout(x,p=0.5,training=self.training)


        # original ---------------------------------
        #prob = F.sigmoid(self.fc(x))

        # config-a
        x0 = x
        prob  = F.sigmoid(self.fc (x0))
        prob1 = F.sigmoid(self.fc1(x0))

        return prob, prob1


def resnext50(**kwargs):
    model = ResNext50(**kwargs)
    return model


def make_new_pretrain():

    src_keys=\
        ['0.weight', '1.weight', '1.bias', '1.running_mean', '1.running_var', '4.0.0.0.0.0.weight', '4.0.0.0.0.1.weight', '4.0.0.0.0.1.bias', '4.0.0.0.0.1.running_mean', '4.0.0.0.0.1.running_var', '4.0.0.0.0.3.weight', '4.0.0.0.0.4.weight', '4.0.0.0.0.4.bias', '4.0.0.0.0.4.running_mean', '4.0.0.0.0.4.running_var', '4.0.0.0.1.weight', '4.0.0.0.2.weight', '4.0.0.0.2.bias', '4.0.0.0.2.running_mean', '4.0.0.0.2.running_var', '4.0.0.1.0.weight', '4.0.0.1.1.weight', '4.0.0.1.1.bias', '4.0.0.1.1.running_mean', '4.0.0.1.1.running_var', '4.1.0.0.0.0.weight', '4.1.0.0.0.1.weight', '4.1.0.0.0.1.bias', '4.1.0.0.0.1.running_mean', '4.1.0.0.0.1.running_var', '4.1.0.0.0.3.weight', '4.1.0.0.0.4.weight', '4.1.0.0.0.4.bias', '4.1.0.0.0.4.running_mean', '4.1.0.0.0.4.running_var', '4.1.0.0.1.weight', '4.1.0.0.2.weight', '4.1.0.0.2.bias', '4.1.0.0.2.running_mean', '4.1.0.0.2.running_var', '4.2.0.0.0.0.weight', '4.2.0.0.0.1.weight', '4.2.0.0.0.1.bias', '4.2.0.0.0.1.running_mean', '4.2.0.0.0.1.running_var', '4.2.0.0.0.3.weight', '4.2.0.0.0.4.weight', '4.2.0.0.0.4.bias', '4.2.0.0.0.4.running_mean', '4.2.0.0.0.4.running_var', '4.2.0.0.1.weight', '4.2.0.0.2.weight', '4.2.0.0.2.bias', '4.2.0.0.2.running_mean', '4.2.0.0.2.running_var', '5.0.0.0.0.0.weight', '5.0.0.0.0.1.weight', '5.0.0.0.0.1.bias', '5.0.0.0.0.1.running_mean', '5.0.0.0.0.1.running_var', '5.0.0.0.0.3.weight', '5.0.0.0.0.4.weight', '5.0.0.0.0.4.bias', '5.0.0.0.0.4.running_mean', '5.0.0.0.0.4.running_var', '5.0.0.0.1.weight', '5.0.0.0.2.weight', '5.0.0.0.2.bias', '5.0.0.0.2.running_mean', '5.0.0.0.2.running_var', '5.0.0.1.0.weight', '5.0.0.1.1.weight', '5.0.0.1.1.bias', '5.0.0.1.1.running_mean', '5.0.0.1.1.running_var', '5.1.0.0.0.0.weight', '5.1.0.0.0.1.weight', '5.1.0.0.0.1.bias', '5.1.0.0.0.1.running_mean', '5.1.0.0.0.1.running_var', '5.1.0.0.0.3.weight', '5.1.0.0.0.4.weight', '5.1.0.0.0.4.bias', '5.1.0.0.0.4.running_mean', '5.1.0.0.0.4.running_var', '5.1.0.0.1.weight', '5.1.0.0.2.weight', '5.1.0.0.2.bias', '5.1.0.0.2.running_mean', '5.1.0.0.2.running_var', '5.2.0.0.0.0.weight', '5.2.0.0.0.1.weight', '5.2.0.0.0.1.bias', '5.2.0.0.0.1.running_mean', '5.2.0.0.0.1.running_var', '5.2.0.0.0.3.weight', '5.2.0.0.0.4.weight', '5.2.0.0.0.4.bias', '5.2.0.0.0.4.running_mean', '5.2.0.0.0.4.running_var', '5.2.0.0.1.weight', '5.2.0.0.2.weight', '5.2.0.0.2.bias', '5.2.0.0.2.running_mean', '5.2.0.0.2.running_var', '5.3.0.0.0.0.weight', '5.3.0.0.0.1.weight', '5.3.0.0.0.1.bias', '5.3.0.0.0.1.running_mean', '5.3.0.0.0.1.running_var', '5.3.0.0.0.3.weight', '5.3.0.0.0.4.weight', '5.3.0.0.0.4.bias', '5.3.0.0.0.4.running_mean', '5.3.0.0.0.4.running_var', '5.3.0.0.1.weight', '5.3.0.0.2.weight', '5.3.0.0.2.bias', '5.3.0.0.2.running_mean', '5.3.0.0.2.running_var', '6.0.0.0.0.0.weight', '6.0.0.0.0.1.weight', '6.0.0.0.0.1.bias', '6.0.0.0.0.1.running_mean', '6.0.0.0.0.1.running_var', '6.0.0.0.0.3.weight', '6.0.0.0.0.4.weight', '6.0.0.0.0.4.bias', '6.0.0.0.0.4.running_mean', '6.0.0.0.0.4.running_var', '6.0.0.0.1.weight', '6.0.0.0.2.weight', '6.0.0.0.2.bias', '6.0.0.0.2.running_mean', '6.0.0.0.2.running_var', '6.0.0.1.0.weight', '6.0.0.1.1.weight', '6.0.0.1.1.bias', '6.0.0.1.1.running_mean', '6.0.0.1.1.running_var', '6.1.0.0.0.0.weight', '6.1.0.0.0.1.weight', '6.1.0.0.0.1.bias', '6.1.0.0.0.1.running_mean', '6.1.0.0.0.1.running_var', '6.1.0.0.0.3.weight', '6.1.0.0.0.4.weight', '6.1.0.0.0.4.bias', '6.1.0.0.0.4.running_mean', '6.1.0.0.0.4.running_var', '6.1.0.0.1.weight', '6.1.0.0.2.weight', '6.1.0.0.2.bias', '6.1.0.0.2.running_mean', '6.1.0.0.2.running_var', '6.2.0.0.0.0.weight', '6.2.0.0.0.1.weight', '6.2.0.0.0.1.bias', '6.2.0.0.0.1.running_mean', '6.2.0.0.0.1.running_var', '6.2.0.0.0.3.weight', '6.2.0.0.0.4.weight', '6.2.0.0.0.4.bias', '6.2.0.0.0.4.running_mean', '6.2.0.0.0.4.running_var', '6.2.0.0.1.weight', '6.2.0.0.2.weight', '6.2.0.0.2.bias', '6.2.0.0.2.running_mean', '6.2.0.0.2.running_var', '6.3.0.0.0.0.weight', '6.3.0.0.0.1.weight', '6.3.0.0.0.1.bias', '6.3.0.0.0.1.running_mean', '6.3.0.0.0.1.running_var', '6.3.0.0.0.3.weight', '6.3.0.0.0.4.weight', '6.3.0.0.0.4.bias', '6.3.0.0.0.4.running_mean', '6.3.0.0.0.4.running_var', '6.3.0.0.1.weight', '6.3.0.0.2.weight', '6.3.0.0.2.bias', '6.3.0.0.2.running_mean', '6.3.0.0.2.running_var', '6.4.0.0.0.0.weight', '6.4.0.0.0.1.weight', '6.4.0.0.0.1.bias', '6.4.0.0.0.1.running_mean', '6.4.0.0.0.1.running_var', '6.4.0.0.0.3.weight', '6.4.0.0.0.4.weight', '6.4.0.0.0.4.bias', '6.4.0.0.0.4.running_mean', '6.4.0.0.0.4.running_var', '6.4.0.0.1.weight', '6.4.0.0.2.weight', '6.4.0.0.2.bias', '6.4.0.0.2.running_mean', '6.4.0.0.2.running_var', '6.5.0.0.0.0.weight', '6.5.0.0.0.1.weight', '6.5.0.0.0.1.bias', '6.5.0.0.0.1.running_mean', '6.5.0.0.0.1.running_var', '6.5.0.0.0.3.weight', '6.5.0.0.0.4.weight', '6.5.0.0.0.4.bias', '6.5.0.0.0.4.running_mean', '6.5.0.0.0.4.running_var', '6.5.0.0.1.weight', '6.5.0.0.2.weight', '6.5.0.0.2.bias', '6.5.0.0.2.running_mean', '6.5.0.0.2.running_var', '7.0.0.0.0.0.weight', '7.0.0.0.0.1.weight', '7.0.0.0.0.1.bias', '7.0.0.0.0.1.running_mean', '7.0.0.0.0.1.running_var', '7.0.0.0.0.3.weight', '7.0.0.0.0.4.weight', '7.0.0.0.0.4.bias', '7.0.0.0.0.4.running_mean', '7.0.0.0.0.4.running_var', '7.0.0.0.1.weight', '7.0.0.0.2.weight', '7.0.0.0.2.bias', '7.0.0.0.2.running_mean', '7.0.0.0.2.running_var', '7.0.0.1.0.weight', '7.0.0.1.1.weight', '7.0.0.1.1.bias', '7.0.0.1.1.running_mean', '7.0.0.1.1.running_var', '7.1.0.0.0.0.weight', '7.1.0.0.0.1.weight', '7.1.0.0.0.1.bias', '7.1.0.0.0.1.running_mean', '7.1.0.0.0.1.running_var', '7.1.0.0.0.3.weight', '7.1.0.0.0.4.weight', '7.1.0.0.0.4.bias', '7.1.0.0.0.4.running_mean', '7.1.0.0.0.4.running_var', '7.1.0.0.1.weight', '7.1.0.0.2.weight', '7.1.0.0.2.bias', '7.1.0.0.2.running_mean', '7.1.0.0.2.running_var', '7.2.0.0.0.0.weight', '7.2.0.0.0.1.weight', '7.2.0.0.0.1.bias', '7.2.0.0.0.1.running_mean', '7.2.0.0.0.1.running_var', '7.2.0.0.0.3.weight', '7.2.0.0.0.4.weight', '7.2.0.0.0.4.bias', '7.2.0.0.0.4.running_mean', '7.2.0.0.0.4.running_var', '7.2.0.0.1.weight', '7.2.0.0.2.weight', '7.2.0.0.2.bias', '7.2.0.0.2.running_mean', '7.2.0.0.2.running_var', '10.1.weight', '10.1.bias']
    dst_keys=\
        ['conv.0.weight', 'conv.1.weight', 'conv.1.bias', 'conv.1.running_mean', 'conv.1.running_var', 'blocks.conv2a.conv.0.weight', 'blocks.conv2a.conv.1.weight', 'blocks.conv2a.conv.1.bias', 'blocks.conv2a.conv.1.running_mean', 'blocks.conv2a.conv.1.running_var', 'blocks.conv2a.conv.3.weight', 'blocks.conv2a.conv.4.weight', 'blocks.conv2a.conv.4.bias', 'blocks.conv2a.conv.4.running_mean', 'blocks.conv2a.conv.4.running_var', 'blocks.conv2a.conv.6.weight', 'blocks.conv2a.conv.7.weight', 'blocks.conv2a.conv.7.bias', 'blocks.conv2a.conv.7.running_mean', 'blocks.conv2a.conv.7.running_var', 'blocks.conv2a.shortcut.0.weight', 'blocks.conv2a.shortcut.1.weight', 'blocks.conv2a.shortcut.1.bias', 'blocks.conv2a.shortcut.1.running_mean', 'blocks.conv2a.shortcut.1.running_var', 'blocks.conv2b.conv.0.weight', 'blocks.conv2b.conv.1.weight', 'blocks.conv2b.conv.1.bias', 'blocks.conv2b.conv.1.running_mean', 'blocks.conv2b.conv.1.running_var', 'blocks.conv2b.conv.3.weight', 'blocks.conv2b.conv.4.weight', 'blocks.conv2b.conv.4.bias', 'blocks.conv2b.conv.4.running_mean', 'blocks.conv2b.conv.4.running_var', 'blocks.conv2b.conv.6.weight', 'blocks.conv2b.conv.7.weight', 'blocks.conv2b.conv.7.bias', 'blocks.conv2b.conv.7.running_mean', 'blocks.conv2b.conv.7.running_var', 'blocks.conv2c.conv.0.weight', 'blocks.conv2c.conv.1.weight', 'blocks.conv2c.conv.1.bias', 'blocks.conv2c.conv.1.running_mean', 'blocks.conv2c.conv.1.running_var', 'blocks.conv2c.conv.3.weight', 'blocks.conv2c.conv.4.weight', 'blocks.conv2c.conv.4.bias', 'blocks.conv2c.conv.4.running_mean', 'blocks.conv2c.conv.4.running_var', 'blocks.conv2c.conv.6.weight', 'blocks.conv2c.conv.7.weight', 'blocks.conv2c.conv.7.bias', 'blocks.conv2c.conv.7.running_mean', 'blocks.conv2c.conv.7.running_var', 'blocks.conv3a.conv.0.weight', 'blocks.conv3a.conv.1.weight', 'blocks.conv3a.conv.1.bias', 'blocks.conv3a.conv.1.running_mean', 'blocks.conv3a.conv.1.running_var', 'blocks.conv3a.conv.3.weight', 'blocks.conv3a.conv.4.weight', 'blocks.conv3a.conv.4.bias', 'blocks.conv3a.conv.4.running_mean', 'blocks.conv3a.conv.4.running_var', 'blocks.conv3a.conv.6.weight', 'blocks.conv3a.conv.7.weight', 'blocks.conv3a.conv.7.bias', 'blocks.conv3a.conv.7.running_mean', 'blocks.conv3a.conv.7.running_var', 'blocks.conv3a.shortcut.0.weight', 'blocks.conv3a.shortcut.1.weight', 'blocks.conv3a.shortcut.1.bias', 'blocks.conv3a.shortcut.1.running_mean', 'blocks.conv3a.shortcut.1.running_var', 'blocks.conv3b.conv.0.weight', 'blocks.conv3b.conv.1.weight', 'blocks.conv3b.conv.1.bias', 'blocks.conv3b.conv.1.running_mean', 'blocks.conv3b.conv.1.running_var', 'blocks.conv3b.conv.3.weight', 'blocks.conv3b.conv.4.weight', 'blocks.conv3b.conv.4.bias', 'blocks.conv3b.conv.4.running_mean', 'blocks.conv3b.conv.4.running_var', 'blocks.conv3b.conv.6.weight', 'blocks.conv3b.conv.7.weight', 'blocks.conv3b.conv.7.bias', 'blocks.conv3b.conv.7.running_mean', 'blocks.conv3b.conv.7.running_var', 'blocks.conv3c.conv.0.weight', 'blocks.conv3c.conv.1.weight', 'blocks.conv3c.conv.1.bias', 'blocks.conv3c.conv.1.running_mean', 'blocks.conv3c.conv.1.running_var', 'blocks.conv3c.conv.3.weight', 'blocks.conv3c.conv.4.weight', 'blocks.conv3c.conv.4.bias', 'blocks.conv3c.conv.4.running_mean', 'blocks.conv3c.conv.4.running_var', 'blocks.conv3c.conv.6.weight', 'blocks.conv3c.conv.7.weight', 'blocks.conv3c.conv.7.bias', 'blocks.conv3c.conv.7.running_mean', 'blocks.conv3c.conv.7.running_var', 'blocks.conv3d.conv.0.weight', 'blocks.conv3d.conv.1.weight', 'blocks.conv3d.conv.1.bias', 'blocks.conv3d.conv.1.running_mean', 'blocks.conv3d.conv.1.running_var', 'blocks.conv3d.conv.3.weight', 'blocks.conv3d.conv.4.weight', 'blocks.conv3d.conv.4.bias', 'blocks.conv3d.conv.4.running_mean', 'blocks.conv3d.conv.4.running_var', 'blocks.conv3d.conv.6.weight', 'blocks.conv3d.conv.7.weight', 'blocks.conv3d.conv.7.bias', 'blocks.conv3d.conv.7.running_mean', 'blocks.conv3d.conv.7.running_var', 'blocks.conv4a.conv.0.weight', 'blocks.conv4a.conv.1.weight', 'blocks.conv4a.conv.1.bias', 'blocks.conv4a.conv.1.running_mean', 'blocks.conv4a.conv.1.running_var', 'blocks.conv4a.conv.3.weight', 'blocks.conv4a.conv.4.weight', 'blocks.conv4a.conv.4.bias', 'blocks.conv4a.conv.4.running_mean', 'blocks.conv4a.conv.4.running_var', 'blocks.conv4a.conv.6.weight', 'blocks.conv4a.conv.7.weight', 'blocks.conv4a.conv.7.bias', 'blocks.conv4a.conv.7.running_mean', 'blocks.conv4a.conv.7.running_var', 'blocks.conv4a.shortcut.0.weight', 'blocks.conv4a.shortcut.1.weight', 'blocks.conv4a.shortcut.1.bias', 'blocks.conv4a.shortcut.1.running_mean', 'blocks.conv4a.shortcut.1.running_var', 'blocks.conv4b.conv.0.weight', 'blocks.conv4b.conv.1.weight', 'blocks.conv4b.conv.1.bias', 'blocks.conv4b.conv.1.running_mean', 'blocks.conv4b.conv.1.running_var', 'blocks.conv4b.conv.3.weight', 'blocks.conv4b.conv.4.weight', 'blocks.conv4b.conv.4.bias', 'blocks.conv4b.conv.4.running_mean', 'blocks.conv4b.conv.4.running_var', 'blocks.conv4b.conv.6.weight', 'blocks.conv4b.conv.7.weight', 'blocks.conv4b.conv.7.bias', 'blocks.conv4b.conv.7.running_mean', 'blocks.conv4b.conv.7.running_var', 'blocks.conv4c.conv.0.weight', 'blocks.conv4c.conv.1.weight', 'blocks.conv4c.conv.1.bias', 'blocks.conv4c.conv.1.running_mean', 'blocks.conv4c.conv.1.running_var', 'blocks.conv4c.conv.3.weight', 'blocks.conv4c.conv.4.weight', 'blocks.conv4c.conv.4.bias', 'blocks.conv4c.conv.4.running_mean', 'blocks.conv4c.conv.4.running_var', 'blocks.conv4c.conv.6.weight', 'blocks.conv4c.conv.7.weight', 'blocks.conv4c.conv.7.bias', 'blocks.conv4c.conv.7.running_mean', 'blocks.conv4c.conv.7.running_var', 'blocks.conv4e.conv.0.weight', 'blocks.conv4e.conv.1.weight', 'blocks.conv4e.conv.1.bias', 'blocks.conv4e.conv.1.running_mean', 'blocks.conv4e.conv.1.running_var', 'blocks.conv4e.conv.3.weight', 'blocks.conv4e.conv.4.weight', 'blocks.conv4e.conv.4.bias', 'blocks.conv4e.conv.4.running_mean', 'blocks.conv4e.conv.4.running_var', 'blocks.conv4e.conv.6.weight', 'blocks.conv4e.conv.7.weight', 'blocks.conv4e.conv.7.bias', 'blocks.conv4e.conv.7.running_mean', 'blocks.conv4e.conv.7.running_var', 'blocks.conv4f.conv.0.weight', 'blocks.conv4f.conv.1.weight', 'blocks.conv4f.conv.1.bias', 'blocks.conv4f.conv.1.running_mean', 'blocks.conv4f.conv.1.running_var', 'blocks.conv4f.conv.3.weight', 'blocks.conv4f.conv.4.weight', 'blocks.conv4f.conv.4.bias', 'blocks.conv4f.conv.4.running_mean', 'blocks.conv4f.conv.4.running_var', 'blocks.conv4f.conv.6.weight', 'blocks.conv4f.conv.7.weight', 'blocks.conv4f.conv.7.bias', 'blocks.conv4f.conv.7.running_mean', 'blocks.conv4f.conv.7.running_var', 'blocks.conv4g.conv.0.weight', 'blocks.conv4g.conv.1.weight', 'blocks.conv4g.conv.1.bias', 'blocks.conv4g.conv.1.running_mean', 'blocks.conv4g.conv.1.running_var', 'blocks.conv4g.conv.3.weight', 'blocks.conv4g.conv.4.weight', 'blocks.conv4g.conv.4.bias', 'blocks.conv4g.conv.4.running_mean', 'blocks.conv4g.conv.4.running_var', 'blocks.conv4g.conv.6.weight', 'blocks.conv4g.conv.7.weight', 'blocks.conv4g.conv.7.bias', 'blocks.conv4g.conv.7.running_mean', 'blocks.conv4g.conv.7.running_var', 'blocks.conv5a.conv.0.weight', 'blocks.conv5a.conv.1.weight', 'blocks.conv5a.conv.1.bias', 'blocks.conv5a.conv.1.running_mean', 'blocks.conv5a.conv.1.running_var', 'blocks.conv5a.conv.3.weight', 'blocks.conv5a.conv.4.weight', 'blocks.conv5a.conv.4.bias', 'blocks.conv5a.conv.4.running_mean', 'blocks.conv5a.conv.4.running_var', 'blocks.conv5a.conv.6.weight', 'blocks.conv5a.conv.7.weight', 'blocks.conv5a.conv.7.bias', 'blocks.conv5a.conv.7.running_mean', 'blocks.conv5a.conv.7.running_var', 'blocks.conv5a.shortcut.0.weight', 'blocks.conv5a.shortcut.1.weight', 'blocks.conv5a.shortcut.1.bias', 'blocks.conv5a.shortcut.1.running_mean', 'blocks.conv5a.shortcut.1.running_var', 'blocks.conv5b.conv.0.weight', 'blocks.conv5b.conv.1.weight', 'blocks.conv5b.conv.1.bias', 'blocks.conv5b.conv.1.running_mean', 'blocks.conv5b.conv.1.running_var', 'blocks.conv5b.conv.3.weight', 'blocks.conv5b.conv.4.weight', 'blocks.conv5b.conv.4.bias', 'blocks.conv5b.conv.4.running_mean', 'blocks.conv5b.conv.4.running_var', 'blocks.conv5b.conv.6.weight', 'blocks.conv5b.conv.7.weight', 'blocks.conv5b.conv.7.bias', 'blocks.conv5b.conv.7.running_mean', 'blocks.conv5b.conv.7.running_var', 'blocks.conv5c.conv.0.weight', 'blocks.conv5c.conv.1.weight', 'blocks.conv5c.conv.1.bias', 'blocks.conv5c.conv.1.running_mean', 'blocks.conv5c.conv.1.running_var', 'blocks.conv5c.conv.3.weight', 'blocks.conv5c.conv.4.weight', 'blocks.conv5c.conv.4.bias', 'blocks.conv5c.conv.4.running_mean', 'blocks.conv5c.conv.4.running_var', 'blocks.conv5c.conv.6.weight', 'blocks.conv5c.conv.7.weight', 'blocks.conv5c.conv.7.bias', 'blocks.conv5c.conv.7.running_mean', 'blocks.conv5c.conv.7.running_var', 'fc.weight', 'fc.bias']


    pretrained_file ='/root/share/project/pytorch/data/pretrain/resnext/resnext_50_32x4d.pth'
    src_dict = torch.load(pretrained_file)
    dst_dict = OrderedDict()
    for k, v in src_dict.items():
        i = src_keys.index(k)
        dst_dict[dst_keys[i]] = v


    new_pretrained_file ='/root/share/project/pytorch/data/pretrain/resnext/new-resnext_50_32x4d.pth'
    torch.save(dst_dict,new_pretrained_file)




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # make_new_pretrain()
    # exit(0)



    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 17 #17
    C,H,W = 3, 256, 256

    #inputs = torch.randn(batch_size,C,H,W)
    #labels = torch.randn(batch_size,num_classes)
    inputs = np.random.uniform(-1.,1.,size=(batch_size,C,H,W)).astype(np.float32)
    labels = np.random.randint(0,num_classes,size=(batch_size,num_classes)).astype(np.float32)
    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels)
    in_shape = inputs.size()[1:]


    net = resnext50(in_shape=in_shape, num_classes=num_classes).cuda().train()
    if 0:
        #pretrained_file ='/root/share/project/pytorch/data/pretrain/resnext/resnext_50_32x4d.pth'
        pretrained_file ='/root/share/project/pytorch/data/pretrain/resnext/new-resnext_50_32x4d.pth'
        skip_list = ['fc.weight', 'fc.bias']  #['10.1.weight', '10.1.bias'] #
        pretrained_dict = torch.load(pretrained_file)
        load_valid(net, pretrained_dict, skip_list=skip_list)

    if 0:
        torch.save( net.state_dict(), '/root/share/project/pytorch/results/xxx.pth')
        ## https://github.com/pytorch/examples/blob/master/ima

    if 0:
        torch.save(net, '/root/share/project/pytorch/results/xxx.torch')

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    probs, probs1 = net.forward(x)
    loss = F.binary_cross_entropy(probs, y)
    loss.backward()

    print(type(net))
    #print(net)

    print('probs')
    print(probs)

    #input('Press ENTER to continue.')

