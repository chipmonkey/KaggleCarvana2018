from common import *

from dataset.box import *
from dataset.annotation import *


def draw_rcnn_pre_nms(img, scores_flat, deltas_flat, rois, cfg, threshold=0.75):

    height,width=img.shape[0:2]
    num_classes = cfg.num_classes

    scores_flat = F.softmax(scores_flat)
    scores = scores_flat.cpu().data.numpy()
    deltas = deltas_flat.cpu().data.numpy()
    deltas = deltas*np.array(cfg.rcnn.train_delta_norm_stds*num_classes)
    rois   = rois.cpu().data.numpy()

    #scores = scores[:,1:]
    num_rois = len(rois)

    labels = np.argmax(scores,axis=1)
    scores = scores[range(0,num_rois),labels]
    inds  = np.argsort(scores)

    for j in range(num_rois):
        i = inds[j]

        s = scores[i]
        l = labels[i]
        if s<threshold or l==0:
            continue

        a = rois[i, 1:5]
        t = deltas[i,l*4:(l+1)*4]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = clip_boxes(b, width, height)  ## clip here if you have drawing error
        b = b.reshape(-1)


        c     =  (s*np.array([0,0,255])).astype(np.uint8)
        color = (int(c[0]),int(c[1]),int(c[2]))
        draw_dotted_rect (img,(a[0], a[1]), (a[2], a[3]), color, 1)

        c     =  (s*np.array(COLORS[l])).astype(np.uint8)   #(s*np.array([0,255,255])).astype(np.uint8)
        color = (int(c[0]),int(c[1]),int(c[2]))
        cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), color, 1)



def draw_rcnn_post_nms(img, dets, threshold=0.8):

    num_classes= len(dets)
    for j in range(1,num_classes): #skip background
        dets_j = dets[j]
        num = len(dets_j)
        for n in range(num):
            s = dets_j[n,4]
            if s<threshold: continue

            b = dets_j[n,0:4]
            c     =  (np.array(COLORS[j])).astype(np.uint8)   #(s*np.array([0,255,255])).astype(np.uint8)
            color = (int(c[0]),int(c[1]),int(c[2]))
            cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), color, 2)

            label=j
            name=NAMES[j]
            text  = '%02d %s : %0.3f'%(label,name,s)
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            textSize = cv2.getTextSize(text, fontFace, fontScale, 2)
            cv2.putText(img, text,(b[0], (int)((b[1] + 2*textSize[1]))), fontFace, fontScale, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, text,(b[0], (int)((b[1] + 2*textSize[1]))), fontFace, fontScale, (255,255,255), 1, cv2.LINE_AA)


#---------------------------------------------------------------------------

#this is in cpu: <todo> change to gpu
def rcnn_nms(x, scores_flat, deltas_flat, resampled_rois, cfg):

    nms_pre_thresh    = cfg.rcnn.test_nms_pre_thresh
    nms_post_thresh   = cfg.rcnn.test_nms_post_thresh
    nms_max_per_image = cfg.rcnn.test_nms_max_per_image
    num_classes =  cfg.num_classes
    # nms_before_thesh = 0.05 ##0.05   # set low numbe r to make roc curve.
                                       # else set high number for faster speed at inference

    height, width = (x.size(2),x.size(3))  #original image width

    scores_flat = F.softmax(scores_flat)
    scores = scores_flat.cpu().data.numpy()

    deltas = deltas_flat.cpu().data.numpy()
    deltas = deltas*np.array(cfg.rcnn.train_delta_norm_stds*num_classes)
    deltas = deltas.reshape(-1,4)

    resampled_rois = resampled_rois.repeat(1,num_classes).cpu().data.numpy().reshape(-1,5)
    proposals = resampled_rois[:, 1:5]
    boxes = box_transform_inv(proposals, deltas)
    boxes = clip_boxes(boxes, width, height)
    boxes = boxes.reshape(-1,4*num_classes)

    #non-max suppression
    dets = [[]for _ in range(num_classes)]
    for j in range(1,num_classes): #skip background
        inds = np.where(scores[:, j] > nms_pre_thresh)[0]

        scores_j = scores[inds, j]
        boxes_j  = boxes [inds, j*4:(j+1)*4]
        dets_j   = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        if len(inds)>0:
            keep   = nms(dets_j, nms_post_thresh)
            dets_j = dets_j[keep, :]

        dets[j] = dets_j


    ##Limit to MAX_PER_IMAGE detections over all classes
    if nms_max_per_image > 0:
        all_scores = np.hstack([dets[j][:, -1] for j in range(1, num_classes)])
        if len(all_scores) > nms_max_per_image:
            all_thresh = np.sort(all_scores)[-nms_max_per_image]
            for j in range(1, num_classes):
                keep = np.where(dets[j][:, -1] >= all_thresh)[0]
                dets[j] = dets[j][keep, :]

    return dets




#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
 