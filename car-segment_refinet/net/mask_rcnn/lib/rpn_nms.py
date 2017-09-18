from common import *

from dataset.box import *
from dataset.annotation import *


def filter_boxes(boxes, min_size):
    '''Remove all boxes with any side smaller than min_size.'''
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def draw_rpn_pre_nms(img, scores_flat, deltas_flat, windows, inside_inds, threshold=0.75):


    height,width=img.shape[0:2]

    scores_flat = F.softmax(scores_flat)
    scores = scores_flat.cpu().data.numpy()
    deltas = deltas_flat.cpu().data.numpy()

    scores = scores[:,1]
    inds  = np.argsort(scores)#[::-1] #sort descend #[::-1]

    num_windows = len(windows)
    insides = np.zeros((num_windows),dtype=np.int32)
    insides[inside_inds]=1
    for j in range(num_windows):
        i = inds[j]
        if insides[i]==0: continue

        s = scores[i]
        if s<threshold:  continue

        a = windows[i]
        t = deltas[i]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = clip_boxes(b, width, height)  ##<todo> clip here if you have drawing error
        b = b.reshape(-1)

        c     =  (s*np.array([0,0,255])).astype(np.uint8)
        color = (int(c[0]),int(c[1]),int(c[2]))
        draw_dotted_rect(img,(a[0], a[1]), (a[2], a[3]),color , 1)
        #cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), (0,v,v), 1)

    for j in range(num_windows):
        i = inds[j]
        if insides[i]==0:
            continue

        a = windows[i]
        t = deltas[i]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = clip_boxes(b, width, height)  ##<todo> clip here if you have drawing error

        b = b.reshape(-1)
        s = scores[i]
        if s<threshold:
            continue

        c     =  (s*np.array([0,255,255])).astype(np.uint8)
        color = (int(c[0]),int(c[1]),int(c[2]))
        #cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (0,0,v), 1)
        cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), color, 1)



def draw_rpn_post_nms(img, rois, roi_scores, top=100):
    roi_scores = roi_scores.cpu().data.numpy()
    rois       = rois.cpu().data.numpy()

    inds = np.argsort(roi_scores)    ##sort descend #[::-1]
    num = len(inds)
    n0 = max(0,num-100)
    for n in range(n0, num):
        i   = inds[n]
        box = rois[i,1:5].astype(np.int)
        v=255*n/num
        color = (0,v,v)
        cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]), color, 1)



#---------------------------------------------------------------------------

#this is in cpu: <todo> change to gpu
def rpn_nms(x, scores_flat, deltas_flat, windows, inside_inds, cfg, mode='train'):

    if mode=='train':
        nms_thresh    = cfg.rpn.train_nms_thresh
        nms_min_size  = cfg.rpn.train_nms_min_size
        nms_pre_topn  = cfg.rpn.train_nms_pre_topn
        nms_post_topn = cfg.rpn.train_nms_post_topn

    elif mode=='eval':
        nms_thresh    = cfg.rpn.test_nms_thresh
        nms_min_size  = cfg.rpn.test_nms_min_size
        nms_pre_topn  = cfg.rpn.test_nms_pre_topn
        nms_post_topn = cfg.rpn.test_nms_post_topn
    else:
        raise ValueError('rpn_nms(): invalid mode = %s?'%mode)

    height, width = (x.size(2),x.size(3))  #original image width

    # 1. Generate proposals from box deltas and windows (shifted bases)
    scores_flat = scores_flat.view(-1, 2, 1) ## ??
    scores_flat = F.softmax(scores_flat)  #[:,1].contiguous()
    scores_flat = scores_flat.cpu().data.numpy()
    deltas_flat = deltas_flat.cpu().data.numpy()

    scores  = scores_flat[inside_inds,1]
    deltas  = deltas_flat[inside_inds]
    windows = windows[inside_inds]

    # Convert anchors into proposals via box transformations
    proposals = box_transform_inv(windows, deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, width, height)

    # 3. remove predicted boxes with either height or width < threshold
    keep      = filter_boxes(proposals, nms_min_size)
    proposals = proposals[keep, :]
    scores    = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if nms_pre_topn > 0:
        order = order[:nms_pre_topn]
        proposals = proposals[order, :]
        scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if nms_post_topn > 0:
        keep = keep[:nms_post_topn]
        proposals = proposals[keep, :]
        scores = scores[keep]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    roi_scores=scores.squeeze()

    num_proposals = len(proposals)
    inds = np.zeros((num_proposals, 1), dtype=np.float32)
    rois = np.hstack((inds, proposals))


    roi_scores = Variable(torch.from_numpy(roi_scores).type(torch.cuda.FloatTensor))
    rois       = Variable(torch.from_numpy(rois).type(torch.cuda.FloatTensor)) #i,x0,y0,x1,y1

    return rois, roi_scores #<todo> modify roi return format later





#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
 
