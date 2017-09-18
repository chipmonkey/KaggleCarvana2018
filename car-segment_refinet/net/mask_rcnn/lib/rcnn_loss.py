#  caffe-fast-rcnn/src/caffe/layers/smooth_L1_loss_layer.cu
#
#  sigma normlisation:
#     https://github.com/rbgirshick/py-faster-rcnn
#        see smooth_l1_loss_param { sigma: 3.0 }
#
#  std normlisation:
#        see cfg.TRAIN.BBOX_NORMALIZE_STDS

from common import *


#---------------------------------------------------------------------------

def rcnn_loss(scores_flat, deltas_flat, rcnn_labels, rcnn_target_inds,  rcnn_targets,  deltas_sigma=3.0):


    num_classes   = scores_flat.size(1)
    rcnn_scores   = scores_flat.view(-1, num_classes)
    rcnn_cls_loss = F.cross_entropy(rcnn_scores, rcnn_labels)

    deltas_sigma2    = deltas_sigma*deltas_sigma
    deltas_flat      = torch.index_select(deltas_flat,0,rcnn_target_inds)
    rcnn_labels      = torch.index_select(rcnn_labels,0,rcnn_target_inds)
    deltas_flat      = deltas_flat.view(-1, 4)
    rcnn_target_inds = Variable(torch.arange(0,rcnn_target_inds.size(0)).type(torch.cuda.LongTensor)*num_classes) + rcnn_labels
    rcnn_deltas      = torch.index_select(deltas_flat, 0, rcnn_target_inds)
    rcnn_reg_loss    = F.smooth_l1_loss( rcnn_deltas*deltas_sigma2, rcnn_targets*deltas_sigma2)*4 /deltas_sigma2

    '''
    http://pytorch.org/docs/0.1.12/_modules/torch/nn/modules/loss.html
    Huber loss
    
    class SmoothL1Loss(_Loss):
    
    
                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
        loss(x, y) = 1/n \sum {
                              { |x_i - y_i| - 0.5,   otherwise
    '''
    return rcnn_cls_loss, rcnn_reg_loss



#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
 