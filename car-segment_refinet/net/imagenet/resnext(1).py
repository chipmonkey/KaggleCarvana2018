# https://github.com/facebookresearch/ResNeXt/blob/master/README.md
# python convert_torch.py -m vgg16.t7

import os
from collections import OrderedDict
from functools import reduce
from net.util import *


#--------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


class ResNext50(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(ResNext50, self).__init__()
        in_channels, height, width = in_shape

        ##resnext_50_32x4d
        self.resnext = nn.Sequential( # Sequential,
            nn.Conv2d(in_channels,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(64,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
        )

        self.fc = nn.Linear(2048, num_classes)

        #nn.Sequential(
        #     Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),
        #     nn.Linear(2048,num_classes)
        # ) # Linear,


    def forward(self, x):

        x = self.resnext(x)

        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = x
        prob  = F.sigmoid(logit)
        return logit,prob

def resnext50(**kwargs):
    model = ResNext50(**kwargs)
    return model

########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 5
    num_classes = 17 #17
    C,H,W = 3, 256, 256

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]


    net = resnext50(in_shape=in_shape, num_classes=num_classes).cuda().train()
    if 1:
        pretrained_file ='/root/share/project/pytorch/data/pretrain/resnext/resnext_50_32x4d.pth'
        skip_list =['fc.weight', 'fc.bias']  #['10.1.weight', '10.1.bias'] #
        pretrained_dict = torch.load(pretrained_file)
        load_valid(net.resnext, pretrained_dict, skip_list=skip_list)

    if 1:
        torch.save({
            'state_dict': net.state_dict(),
        },  '/root/share/project/pytorch/results/xxx.pth')
        ## https://github.com/pytorch/examples/blob/master/ima

    if 0:
        torch.save(net, '/root/share/project/pytorch/results/xxx.torch')

    x = Variable(inputs)
    logits, probs = net.forward(x.cuda())

    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels.cuda()))
    loss.backward()

    print(type(net))
    #print(net)

    print('probs')
    print(probs)

    #input('Press ENTER to continue.')

