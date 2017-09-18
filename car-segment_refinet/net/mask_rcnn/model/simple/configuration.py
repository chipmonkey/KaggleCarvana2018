from common import *
#import json
import pprint as pprint

#configuration
class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()

        #net
        self.pool_size   = 7
        self.num_classes = 3

        #image
        self.train_max_size = 1280
        #self.train.images_per_batch  = 1

        #rpn
        self.rpn = types.SimpleNamespace()
        self.rpn.scales=(8, 16, 32)
        self.rpn.ratios=(0.5, 1, 2)
        self.rpn.stride=16
        self.rpn.allowed_border= -1

        self.rpn.train_batch_size     = 256  # rpn target
        self.rpn.train_fg_fraction    = 0.5
        self.rpn.train_bg_thresh_high = 0.7
        self.rpn.train_fg_thresh_low  = 0.3

        self.rpn.train_nms_thresh    = 0.7 # rpn nms
        self.rpn.train_nms_min_size  = 8
        self.rpn.train_nms_pre_topn  = 12000
        self.rpn.train_nms_post_topn =  6000

        self.rpn.test_nms_thresh    = self.rpn.train_nms_thresh
        self.rpn.test_nms_min_size  = self.rpn.train_nms_min_size
        self.rpn.test_nms_pre_topn  = 500
        self.rpn.test_nms_post_topn = 100


        #rcnn
        self.rcnn = types.SimpleNamespace()
        self.rcnn.train_batch_size      = 128  # rcnn target
        self.rcnn.train_fg_fraction     = 0.25
        self.rcnn.train_bg_thresh_high  = 0.5
        self.rcnn.train_bg_thresh_low   = 0.1
        self.rcnn.train_fg_thresh_low   = 0.5
        self.rcnn.train_delta_norm_stds = (0.1, 0.1, 0.2, 0.2)

        self.rcnn.test_nms_pre_thresh    = 0.5 # set low 0.05 to make roc curve.
        self.rcnn.test_nms_post_thresh   = 0.3
        self.rcnn.test_nms_max_per_image = 100

        #fcn (mask segmentation)
        self.fcn = types.SimpleNamespace()
        self.fcn.enlarge = 2

    def check(self,feature_net = None, fcn_net= None):
        if feature_net is not None:
            assert(feature_net.stride == self.rpn.stride)

        if fcn_net is not None:
            assert(fcn_net.enlarge == self.fcn.enlarge)

    def save(self, file):
        d    = self.__dict__.copy()
        rpn  = d.pop('rpn',None)
        rcnn = d.pop('rcnn',None)
        fcn  = d.pop('fcn',None)
        rpn  = vars(rpn)
        rcnn = vars(rcnn)
        fcn  = vars(fcn)

        # with open(file, 'w') as f:
        #     json.dump(d, f, indent=4)
        #     f.write('hi there\n')
        with open(file, 'w') as f:
            pprint.pprint(d, width=1,stream=f)
            f.write('\n#rpn\n')
            pprint.pprint(rpn, width=80,stream=f)
            f.write('\n#rcnn\n')
            pprint.pprint(rcnn, width=80,stream=f)
            f.write('\n#fcn\n')
            pprint.pprint(fcn, width=80,stream=f)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    cfg = Configuration()
    cfg.save('/root/share/project/lung-cancer/results/configure')

