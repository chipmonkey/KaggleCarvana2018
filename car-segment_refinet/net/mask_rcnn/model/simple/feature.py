from common import*


#----- helper functions --------

def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class FeatureNet(nn.Module):

    def  __init__(self, in_shape=(3,256,256), out_channels=32):
        super(FeatureNet, self).__init__()
        in_channels, height, width = in_shape
        self.stride = 16

        self.layer0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
        )
        self.layer1 = nn.Sequential(
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        self.layer2 = nn.Sequential(
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        self.layer3 = nn.Sequential(
            *make_conv_bn_relu(16, out_channels, kernel_size=3, stride=1, padding=1 ),
        )


    def forward(self, x):
        x = self.layer0(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # stride=2
        x = self.layer1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # stride=4
        x = self.layer2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # stride=8
        x = self.layer3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # stride=16

        feature = x
        return feature



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    batch_size = 1
    channel, height, width = 3,256,256
    out_channels = 32

    inputs      = torch.randn(batch_size,channel,height,width)
    feature_net = FeatureNet((channel,height,width), out_channels).cuda().train()

    #check
    x = Variable(inputs).cuda()
    f = feature_net(x)

    print(type(feature_net))
    print(feature_net)
    print('feature\n',f)

    #check stride, etc
    print('feature_net.stride\n', feature_net.stride,'\n')
    print('inputs.size()\n', inputs.size(),'\n')
    print('f.size()\n', f.size(),'\n')

