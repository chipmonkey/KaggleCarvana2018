from common import*
from net.mask_rcnn.model.simple.configuration import *



def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

#for segmentation
# see paper 'Mask R-CNN' - Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick (Facebook AI Research (FAIR))
# figure (3)
#

class FcnNet(nn.Module):

    def __init__(self, cfg, in_channels ):
        super(FcnNet, self).__init__()
        num_classes = cfg.num_classes
        self.enlarge = 2

        self.layer0 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            *make_conv_bn_relu(in_channels, 256, kernel_size=3, stride=1, padding=1 ),
        )
        self.predict_mask =  nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, groups=1, bias=True)


    def forward(self, x):
        x = self.layer0(x)
        masks = self.predict_mask(x)

        return masks


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    cfg = Configuration()
    if 1:  #modify default configurations here
        cfg.fcn.enlarge=2

    enlarge = cfg.fcn.enlarge
    in_channels = 32
    num_classes = 3
    pool_size = 7
    num_rois = 10

    inputs = torch.randn(num_rois, in_channels, pool_size, pool_size)

    fcn_net = FcnNet(cfg, in_channels ).cuda().train()
    x = Variable(inputs).cuda()
    masks = fcn_net(x)

    print(type(fcn_net))
    print(fcn_net)
    print('masks\n',masks,'\n')
    print('masks.size()\n',masks.size(),'\n')
    print('num_rois\n',num_rois,'\n')
    print('num_classes\n',num_classes,'\n')
    print('pool_size\n',pool_size,'\n')
    print('enlarge\n',enlarge,'\n')