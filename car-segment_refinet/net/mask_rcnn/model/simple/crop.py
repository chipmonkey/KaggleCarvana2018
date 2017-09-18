from common import*
from net.mask_rcnn.model.simple.configuration import *

from net.mask_rcnn.lib.roi_align_pool.module import RoIAlignMax as Crop
# from net.mask_rcnn.lib.roi_pool.module import RoIPool as Crop


#----- helper functions --------

def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class CropNet(nn.Module):

    def __init__(self, cfg, in_channels, out_channels):
        super(CropNet, self).__init__()
        stride    = cfg.rpn.stride
        pool_size = cfg.pool_size


        self.crop = Crop(pool_size, pool_size, 1./stride)
        self.layer0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
        )
        self.layer1 = nn.Sequential(
            *make_conv_bn_relu(16, out_channels, kernel_size=1, stride=1, padding=0 ),
        )


    def forward(self, x,sampled_rois):
        x = self.crop(x,sampled_rois)
        x = self.layer0(x)
        x = self.layer1(x)

        return x



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    cfg = Configuration() #default configuration
    cfg.pool_size  = 7
    cfg.rpn.stride = 16

    batch_size   = 1
    in_channels  = 16
    out_channels = 32
    width,height = 800, 400

    stride = cfg.rpn.stride
    H,W = height//stride, width//stride

    #make dummy rois
    num_rois = 10
    rois_data = np.zeros((num_rois,5),np.float32)
    for n in range(num_rois):
        x0 = random.randrange(0, width -20)
        y0 = random.randrange(0, height-20)
        x1 = random.randrange(x0, width )
        y1 = random.randrange(y0, height)
        rois_data[n,1] = x0
        rois_data[n,2] = y0
        rois_data[n,3] = x1
        rois_data[n,4] = y1
    rois = torch.from_numpy(rois_data)

    #make dummy inputs
    inputs = torch.randn(batch_size,in_channels,H,W)

    #check
    x    = Variable(inputs).cuda()
    rois = Variable(rois).cuda()

    crop_net = CropNet(cfg, in_channels, out_channels).cuda().train()
    crops = crop_net(x, rois)

    print(type(crop_net))
    print(crop_net)
    print('crops\n',crops)
    print('crops.size()\n', crops.size(),'\n')
    print('num_rois\n', num_rois,'\n')

