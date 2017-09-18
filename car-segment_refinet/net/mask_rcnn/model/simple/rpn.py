from common import*
from dataset.box import *
from net.mask_rcnn.model.simple.configuration  import *

def rpn_bases(cfg):
    bases = make_bases(ratios = np.array(cfg.rpn.ratios), scales = np.array(cfg.rpn.scales))
    return bases

def rpn_windows(x, f, bases, cfg):

    stride         = cfg.rpn.stride
    allowed_border = cfg.rpn.allowed_border
    image_shape    = (x.size(2),x.size(3))  #original image width
    feature_shape  = (f.size(2),f.size(3))
    windows, inside_inds = make_windows(bases, stride, image_shape, feature_shape, allowed_border)

    return windows, inside_inds


class RpnNet(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RpnNet, self).__init__()
        self.cfg = cfg
        num_bases = len(cfg.rpn.ratios)*len(cfg.rpn.scales)


        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.predict_score =  nn.Conv2d(256, num_bases*2, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.predict_dbox  =  nn.Conv2d(256, num_bases*4, kernel_size=1, stride=1, padding=0, groups=1, bias=True)


    def forward(self, x):

        x     = self.conv(x)
        delta = self.predict_dbox (x)
        score = self.predict_score(x)
        delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        score_flat = score.permute(0, 2, 3, 1).contiguous().view(-1, 2)

        #loss takes in logit!
        return score_flat, delta_flat


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    cfg = Configuration()
    if 1:  #modify default configurations here
        cfg.rpn.train_fg_thresh_low = 0.7
        cfg.rpn.scales=[64,128,256]
        cfg.rpn.ratios=[1,0.5]

    batch_size = 1
    in_channels, H, W  = 32, 256,256

    inputs  = torch.randn(batch_size,in_channels,H,W)
    rpn_net = RpnNet(cfg, in_channels ).cuda().train()

    x = Variable(inputs).cuda()
    s,b = rpn_net(x)

    print(type(rpn_net))
    print(rpn_net)
    print('score_flat\n',s)
    print('delta_flat\n',b)