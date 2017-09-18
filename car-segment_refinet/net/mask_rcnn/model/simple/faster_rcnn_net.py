from common import*

from net.mask_rcnn.lib.rcnn_nms import *
from net.mask_rcnn.lib.rcnn_target import *
from net.mask_rcnn.lib.rcnn_loss import *
from net.mask_rcnn.lib.rpn_nms import *
from net.mask_rcnn.lib.rpn_target import *
from net.mask_rcnn.lib.rpn_loss import *

from net.mask_rcnn.model.simple.configuration  import *
from net.mask_rcnn.model.simple.feature  import FeatureNet
from net.mask_rcnn.model.simple.rpn      import RpnNet, rpn_bases, rpn_windows
from net.mask_rcnn.model.simple.crop     import CropNet
from net.mask_rcnn.model.simple.rcnn     import RcnnNet
from net.mask_rcnn.model.simple.fcn      import FcnNet


#---------------------------------------------------------------------------

# https://github.com/longcw/faster_rcnn_pytorch
# https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/faster_rcnn.py


class FasterRcnnNet(object):
    mode = 'train'

    def __init__(self, cfg):
        super(FasterRcnnNet, self).__init__()
        self.cfg  = cfg
        pool_size   = cfg.pool_size
        num_classes = cfg.num_classes
        stride      = cfg.rpn.stride

        # inference
        self.feature_net = FeatureNet(out_channels=32).cuda()
        self.rpn_net  = RpnNet (cfg, in_channels=32).cuda()
        self.crop_net = CropNet(cfg, in_channels=32, out_channels=24).cuda()
        self.rcnn_net = RcnnNet(cfg, in_channels=24).cuda()
        self.fcn_net  = FcnNet (cfg, in_channels=24).cuda()

        # check --
        self.bases = rpn_bases(cfg)
        self.cfg.check(feature_net = self.feature_net, fcn_net = self.fcn_net)

        # extract list of sub-net, etc ...
        d = self.__dict__
        d = { k : d[k] for k,v in d.items() if '_net' in k }
        names  =  list(d.keys())
        nets   =  list(d.values())
        l      = [ list(net.parameters()) for net in nets    ]
        params = [ item for sublist in l for item in sublist ]

        self.names  = names
        self.nets   = nets
        self.params = params


    def forward(self, x, annotation=None):

        if self.mode=='train':
            for net in self.nets :
                net.train()

        elif self.mode=='eval':
            for net in self.nets :
                net.eval()

        else:
            raise ValueError('forward: invalid mode = %s?'%self.model)


        cfg   = self.cfg
        mode  = self.mode
        bases = self.bases

        f             = self.feature_net(x)
        rpn_s, rpn_d  = self.rpn_net(f)
        windows, inside_inds = rpn_windows(x, f, bases, cfg)
        rois, roi_scores     = rpn_nms(x, rpn_s, rpn_d, windows, inside_inds, cfg, mode)

        if mode=='train':
            rpn_label_inds, rpn_labels, rpn_target_inds, rpn_targets  = rpn_target(windows, inside_inds, annotation, cfg)
            sampled_rois, rcnn_labels, rcnn_target_inds, rcnn_targets = rcnn_target(rois, annotation, cfg)

        elif mode=='eval':
            sampled_rois   = rois
            rpn_label_inds, rpn_labels, rpn_target_inds, rpn_targets  = None, None, None, None,
            rcnn_labels, rcnn_target_inds, rcnn_targets = None, None, None

        crops          = self.crop_net(f,sampled_rois)
        rcnn_s, rcnn_d = self.rcnn_net(crops)
        masks          = self.fcn_net(crops)
        dets           = rcnn_nms(x, rcnn_s, rcnn_d, sampled_rois, cfg)



        self.inputs   = x
        self.features = f

        self.rpn_scores_flat  = rpn_s
        self.rpn_deltas_flat  = rpn_d
        self.rcnn_scores_flat = rcnn_s
        self.rcnn_deltas_flat = rcnn_d

        self.windows     = windows
        self.inside_inds = inside_inds
        self.rois           = rois
        self.roi_scores     = roi_scores
        self.sampled_rois   = sampled_rois
        self.dets  = dets
        self.masks = masks

        self.rpn_label_inds   = rpn_label_inds
        self.rpn_labels       = rpn_labels
        self.rpn_target_inds  = rpn_target_inds
        self.rpn_targets      = rpn_targets
        self.rcnn_labels      = rcnn_labels
        self.rcnn_target_inds = rcnn_target_inds
        self.rcnn_targets     = rcnn_targets

        return dets



    def loss(self, x, annotation):
        if self.mode=='train':#this is always in training mode!
            cfg = self.cfg

            rpn_scores_flat  = self.rpn_scores_flat
            rpn_deltas_flat  = self.rpn_deltas_flat
            rcnn_scores_flat = self.rcnn_scores_flat
            rcnn_deltas_flat = self.rcnn_deltas_flat

            rpn_label_inds   = self.rpn_label_inds
            rpn_labels       = self.rpn_labels
            rpn_target_inds  = self.rpn_target_inds
            rpn_targets      = self.rpn_targets
            rcnn_labels      = self.rcnn_labels
            rcnn_target_inds = self.rcnn_target_inds
            rcnn_targets     = self.rcnn_targets

            rpn_cls_loss, rpn_reg_loss  = rpn_loss(rpn_scores_flat, rpn_deltas_flat, rpn_label_inds, rpn_labels, rpn_target_inds, rpn_targets)
            rcnn_cls_loss, rcnn_reg_loss = rcnn_loss(rcnn_scores_flat, rcnn_deltas_flat, rcnn_labels, rcnn_target_inds, rcnn_targets)
            total_loss = rpn_cls_loss + rpn_reg_loss  + rcnn_cls_loss + rcnn_reg_loss

            self.rpn_cls_loss  = rpn_cls_loss
            self.rpn_reg_loss  = rpn_reg_loss
            self.rcnn_cls_loss = rcnn_cls_loss
            self.rcnn_reg_loss = rcnn_reg_loss

            return total_loss

        else:
            raise ValueError('loss: invalid mode = %s?'%self.model)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)

        #save header
        with open(model_dir +'/faster_rcnn_net.txt', 'w') as f:
            f.write('%s\n\n'%str(type(self )))
            f.write(inspect.getsource(self.__init__)+'\n')
            f.write(inspect.getsource(self.forward )+'\n')
            f.write('%s\n\n'%('-'*100))
            f.write(str(self))
            f.write('\n')
        self.cfg.save(model_dir +'/cfg')
        np.savetxt(model_dir +'/bases.txt',self.bases,fmt='%0.1f')

        #save each sub-net
        state_dicts = {}
        num_nets   = len(self.nets)
        for n in range(num_nets):
            net  = self.nets[n]
            name = self.names[n]
            state_dicts[name] = net.state_dict()

            with open(model_dir +'/%s.txt'%name, 'w') as f:
                f.write('%s\n\n'%str(type(net )))
                f.write(inspect.getsource(net.__init__)+'\n')
                f.write(inspect.getsource(net.forward )+'\n')
                f.write('%s\n\n'%('-'*100))
                f.write(str(net))
                f.write('\n')

        torch.save(state_dicts, model_dir +'/state_dics.pth')

    def load(self, model_dir):

        state_dicts = torch.load( model_dir +'/state_dics.pth')

        num_nets   = len(self.nets)
        for n in range(num_nets):
            net  = self.nets[n]
            name = self.names[n]

            net.load_state_dict(state_dicts[name])




#-----------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #dummy data
    annotation_file = '/media/ssd/[data]/dummy/object-det-debug/bird_and_flower/annotations/000.txt'
    img_file        = '/media/ssd/[data]/dummy/object-det-debug/bird_and_flower/images/000.jpg'

    annotation = load_annotation(annotation_file, img_file)
    img        = cv2.imread(img_file)
    x, a = default_transform(img, annotation)


    #parameters
    cfg = Configuration() #default configuration

    #change configurations here ...
    cfg.rpn.train_fg_thresh_low = 0.8
    cfg.rpn.scales=[64,128,256]
    cfg.rpn.ratios=[1,0.5]

    faster_rcnn_net = FasterRcnnNet(cfg)
    x = Variable(x).cuda()

    faster_rcnn_net.mode = 'train'
    faster_rcnn_net.forward(x, annotation)


    faster_rcnn_net.mode = 'eval'
    faster_rcnn_net.forward(x)