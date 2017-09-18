from common import*
from net.mask_rcnn.model.simple.configuration import *


def make_fc_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]

def make_flat(out):
    flat = out.view(out.size(0), -1)
    return flat


class RcnnNet(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RcnnNet, self).__init__()
        pool_size = cfg.pool_size
        num_classes = cfg.num_classes

        # CNN
        self.linear = nn.Sequential(
            *make_fc_bn_relu(in_channels*pool_size*pool_size,4096),
            *make_fc_bn_relu(4096,4096),
        )
        self.predict_score =  nn.Linear(4096, num_classes,   bias=True)
        self.predict_dbox  =  nn.Linear(4096, num_classes*4, bias=True)


    def forward(self, x):

        x = make_flat(x)
        x = self.linear(x)
        #x = F.dropout(x, p=0.5,training=self.training)

        deltas_flat = self.predict_dbox (x)
        scores_flat = self.predict_score(x)

        #loss takes in logit!
        return scores_flat, deltas_flat


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    cfg = Configuration()
    if 1:  #modify default configurations here
        cfg.num_classes  = 3
        cfg.pool_size    = 7

    num_rois   = 100
    in_channels = 32
    pool_size   = cfg.pool_size

    inputs = torch.randn(num_rois, in_channels, pool_size, pool_size)
    inputs = Variable(inputs).cuda()

    rcnn_net = RcnnNet(cfg, in_channels).cuda().train()
    s,b = rcnn_net(inputs)

    print(type(rcnn_net))
    print(rcnn_net)
    print('score_flat\n',s)
    print('delta_flat\n',b)