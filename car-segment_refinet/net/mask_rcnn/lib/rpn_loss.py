#  caffe-fast-rcnn/src/caffe/layers/smooth_L1_loss_layer.cu
#
#  sigma normlisation:
#     https://github.com/rbgirshick/py-faster-rcnn
#        see smooth_l1_loss_param { sigma: 3.0 }
#
#  std normlisation:
#        see cfg.TRAIN.BBOX_NORMALIZE_STDS

from common import *


def modified_smooth_l1( box_preds, box_targets, sigma=3.0):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise

        inside_weights  = 1
        outside_weights = 1/num_examples
    '''
    sigma2 = sigma * sigma
    diffs  =  box_preds-box_targets
    smooth_l1_signs = torch.abs(diffs) <  (1.0 / sigma2)
    smooth_l1_signs = smooth_l1_signs.type(torch.cuda.FloatTensor)

    smooth_l1_option1 = 0.5 * diffs* diffs *  sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5  / sigma2
    smooth_l1 = smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs)
    l1 = smooth_l1.sum()/len(smooth_l1)

    return l1


#---------------------------------------------------------------------------

def rpn_loss(scores_flat, deltas_flat, rpn_label_inds, rpn_labels, rpn_target_inds, rpn_targets,  deltas_sigma=3.0):

    rpn_scores    = torch.index_select(scores_flat,0,rpn_label_inds)
    rpn_cls_loss  = F.cross_entropy(rpn_scores, rpn_labels)

    deltas_sigma2 = deltas_sigma*deltas_sigma
    rpn_deltas    = torch.index_select(deltas_flat,0,rpn_target_inds)
    rpn_reg_loss  = F.smooth_l1_loss( rpn_deltas*deltas_sigma2, rpn_targets*deltas_sigma2)*4 /deltas_sigma2

    '''
    http://pytorch.org/docs/0.1.12/_modules/torch/nn/modules/loss.html
    Huber loss
    
    class SmoothL1Loss(_Loss):
    
    
                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
        loss(x, y) = 1/n \sum {
                              { |x_i - y_i| - 0.5,   otherwise
    '''
    #debug (cross check)
    #l = modified_smooth_l1(rpn_deltas, rpn_targets, deltas_sigma)

    return rpn_cls_loss, rpn_reg_loss




def check_layer():

    # set some dummy data
    H = 5
    W = 4
    num_bases  = 3
    batch_size = 1
    L=8

    scores_data = np.random.uniform(-1.,1.,(batch_size,num_bases*2,H,W))
    deltas_data = np.random.uniform(-2.,2.,(batch_size,num_bases*4,H,W))

    rpn_labels_data     = np.random.choice([0,1],L)
    rpn_label_inds_data = np.random.choice(np.arange(H*W*num_bases),L, replace=False)

    rpn_target_inds_data = rpn_label_inds_data[np.where(rpn_labels_data==1)[0]]
    rpn_targets_data     = np.random.uniform(-2.,2.,(len(rpn_target_inds_data),4))

    scores = Variable(torch.from_numpy(scores_data).type(torch.FloatTensor)).cuda()
    deltas = Variable(torch.from_numpy(deltas_data).type(torch.FloatTensor)).cuda()
    scores_flat = scores.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    deltas_flat = deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)

    rpn_label_inds  = Variable(torch.from_numpy(rpn_label_inds_data).type(torch.cuda.LongTensor))
    rpn_labels      = Variable(torch.from_numpy(rpn_labels_data).type(torch.cuda.LongTensor))
    rpn_target_inds = Variable(torch.from_numpy(rpn_target_inds_data).type(torch.cuda.LongTensor))
    rpn_targets     = Variable(torch.from_numpy(rpn_targets_data).type(torch.cuda.FloatTensor))



    # check layer
    rpn_cls_loss, rpn_reg_loss = rpn_loss(scores_flat, deltas_flat, rpn_label_inds, rpn_labels, rpn_target_inds, rpn_targets)


    print(rpn_cls_loss)
    print(rpn_reg_loss)
    pass

#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 