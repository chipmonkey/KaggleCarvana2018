# mask-faster-rcnn

from dataset.annotation import *
from dataset.box import *


# rcnn target
def rcnn_compute_targets(et_boxes, gt_boxes, labels):

    assert et_boxes.shape[0] == gt_boxes.shape[0]
    assert et_boxes.shape[1] == 4
    assert gt_boxes.shape[1] == 4
    targets = box_transform(et_boxes, gt_boxes).astype(np.float32, copy=False)

    return targets



# for debug (draw results of ground truth generation)-------------------------
def draw_rois(img, rois, roi_scores):
    roi_scores = roi_scores.cpu().data.numpy()
    rois       = rois.cpu().data.numpy()

    inds = np.argsort(roi_scores)       #sort ascend #[::-1]
    num = len(inds)
    for n in range(0, num):
        i   = inds[n]
        box = rois[i,1:5].astype(np.int)
        v=255*n/num
        color = (0,v,v)
        cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]), color, 1)



def draw_rcnn_labels(img, rois, labels, is_fg=1, is_bg=1, is_print=1):
    rois    = rois.cpu().data.numpy()
    labels  = labels.cpu().data.numpy()

    ## draw +ve/-ve labels ......
    boxes  = rois[:,1:5].astype(np.int32)
    labels = labels.reshape(-1)

    fg_label_inds = np.where(labels != 0)[0]
    bg_label_inds = np.where(labels == 0)[0]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print: print ('rcnn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))


    if is_bg:
        for i in bg_label_inds:
            a = boxes[i]
            cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (32,32,32), 1)
            cv2.circle(img,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (32,32,32), -1, cv2.LINE_AA)

    if is_fg:
        for i in fg_label_inds:
            color = COLORS[labels[i]]  #(0,0,255) #
            a = boxes[i]
            cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), color, 1)
            cv2.circle(img,((a[0]+a[2])//2, (a[1]+a[3])//2),2, color, -1, cv2.LINE_AA)



def draw_rcnn_targets(img, rois, target_inds,  targets,  cfg, is_before=1, is_after=1, is_print=1):
    rois = rois.cpu().data.numpy()
    target_inds = target_inds.cpu().data.numpy()
    targets     = targets.cpu().data.numpy()

    #draw +ve targets ......
    boxes = rois[:,1:5].astype(np.int32)


    num_pos_target = len(target_inds)
    if is_print:
        print ('rcnn target : num_pos=%d'  %(num_pos_target))


    for n,i in enumerate(target_inds):
        a = boxes[i]
        t = targets[n]*np.array(cfg.rcnn.train_delta_norm_stds)
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        if is_before:
            cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
            cv2.circle(img,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (0,0,255), -1, cv2.LINE_AA)

        if is_after:
            cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), (0,255,255), 1)





# Faster-rcnn ground-truth layer rcnn----------------------------------------
#<todo> change to gpu
def rcnn_target(rois, annotation, cfg):

    num_classes = cfg.num_classes
    gt_boxes  = annotation['boxes' ]
    gt_labels = annotation['labels']

    rois = rois.view(-1,5)  # Proposal (i, x0, y0, x1, y1) coming from RPN
    rois = rois.cpu().data.numpy()

    num_gt_boxes = len(gt_boxes)
    zeros        = np.zeros((num_gt_boxes, 1), dtype=np.float32)
    rois         = np.vstack((rois, np.hstack((zeros, gt_boxes)))) #gtbox is always used for training
    assert np.all(rois[:, 0] == 0), 'Only single image batches are supported'

    rois_per_image    = cfg.rcnn.train_batch_size
    fg_rois_per_image = np.round(cfg.rcnn.train_fg_fraction * rois_per_image)

    # overlaps: (rois x gt_boxes)
    boxes = rois[:,1:5]
    overlaps = box_overlap(
        np.ascontiguousarray(boxes,    dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.rcnn.train_fg_thresh_low)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps <  cfg.rcnn.train_bg_thresh_high ) &
                       (max_overlaps >= cfg.rcnn.train_bg_thresh_low  ))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep_inds      = np.append(fg_inds, bg_inds)
    resampled_rois = rois[keep_inds]

    labels = labels[keep_inds]
    labels[fg_rois_per_this_image:] = 0  # Clamp labels for the background RoIs to 0


    idx = np.where(labels !=  0)[0]
    target_inds = idx
    gt_boxes = gt_boxes[gt_assignment[keep_inds[idx]], :]
    et_boxes = resampled_rois[idx, 1:5]
    targets  = rcnn_compute_targets(et_boxes, gt_boxes,  labels)
    targets  = targets / np.array(cfg.rcnn.train_delta_norm_stds)  # this is for each box

    resampled_rois  = Variable(torch.from_numpy(resampled_rois).type(torch.cuda.FloatTensor))
    labels      = Variable(torch.from_numpy(labels).type(torch.cuda.LongTensor))
    target_inds = Variable(torch.from_numpy(target_inds).type(torch.cuda.LongTensor))
    targets     = Variable(torch.from_numpy(targets).type(torch.cuda.FloatTensor))

    return resampled_rois, labels, target_inds, targets





def check_layer():

    annotation_file = '/media/ssd/[data]/dummy/object-det-debug/bird_and_flower/annotations/000.txt'
    img_file        = '/media/ssd/[data]/dummy/object-det-debug/bird_and_flower/images/000.jpg'
    # annotation_file = '/media/ssd/[data]/dummy/object-det-debug/rose/annotations/000.txt'
    # img_file        = '/media/ssd/[data]/dummy/object-det-debug/rose/images/000.jpg'


    # set some dummy data
    annotation = load_annotation(annotation_file, img_file)
    img        = cv2.imread(img_file)

    height, width  = img.shape[0:2]
    gt_boxes  = annotation['boxes']
    num_gt_boxes = len(gt_boxes)

    N=100
    # roi_scores_data0 = np.random.uniform(0,0.8,N)
    roi_scores_data0 = np.zeros(N,np.float32)
    rois_data0 = np.zeros((N,5),np.float32)
    for l in range(N):
        gt_box = gt_boxes[random.randint(0,num_gt_boxes-1)]
        cx = (gt_box[2]+gt_box[0])//2
        cy = (gt_box[3]+gt_box[1])//2
        w  = (gt_box[2]-gt_box[0])
        h  = (gt_box[3]-gt_box[1])

        cx += random.randint(0,int(1.5*w))
        cy += random.randint(0,int(1.5*h))
        w = int(random.uniform(0.3,1.5)*w)
        h = int(random.uniform(0.3,1.5)*h)
        rois_data0[l,1]=cx-w//2
        rois_data0[l,2]=cy-h//2
        rois_data0[l,3]=cx+w//2
        rois_data0[l,4]=cy+h//2

        roi_scores_data0[l] = one_box_overlap(rois_data0[l,1:5],gt_box)
        pass

    P=50
    # roi_scores_data1 = np.random.uniform(0.8,1,P)
    roi_scores_data1 = np.zeros(P,np.float32)
    rois_data1 = np.zeros((P,5),np.float32)
    for l in range(P):
        gt_box = gt_boxes[random.randint(0,num_gt_boxes-1)]
        cx = (gt_box[2]+gt_box[0])//2
        cy = (gt_box[3]+gt_box[1])//2
        w  = (gt_box[2]-gt_box[0])
        h  = (gt_box[3]-gt_box[1])

        cx += random.randint(0,w//4)
        cy += random.randint(0,h//4)
        w = int(random.uniform(0.9,1.2)*w)
        h = int(random.uniform(0.9,1.2)*h)
        rois_data1[l,1]=cx-w//2
        rois_data1[l,2]=cy-h//2
        rois_data1[l,3]=cx+w//2
        rois_data1[l,4]=cy+h//2

        roi_scores_data1[l] = one_box_overlap(rois_data1[l,1:5],gt_box)
        pass
    rois_data = np.vstack((rois_data0,rois_data1))
    roi_scores_data = np.concatenate((roi_scores_data0,roi_scores_data1))




    roi_scores = Variable(torch.from_numpy(roi_scores_data).type(torch.cuda.FloatTensor))
    rois       = Variable(torch.from_numpy(rois_data).type(torch.cuda.FloatTensor))

    # check layer
    cfg = Configuration() #default configuration
    cfg.rcnn.train_batch_size      = 128
    cfg.rcnn.train_fg_fraction     = 0.25
    cfg.rcnn.train_bg_thresh_high  = 0.5
    cfg.rcnn.train_bg_thresh_low   = 0.1
    cfg.rcnn.train_fg_thresh_low   = 0.5
    cfg.rcnn.train_delta_norm_stds = (0.1, 0.1, 0.2, 0.2)


    resampled_rois, labels, target_inds, targets = rcnn_target(rois, annotation, cfg)

    #check
    gt_boxes  = annotation['boxes' ]
    gt_labels = annotation['labels']

    #draw_rois(img, rois, roi_scores)
    #im_show('img1',img)
    #draw_gt_boxes(img, gt_boxes, gt_labels)
    #draw_rcnn_labels(img, resampled_rois,  labels, is_fg=1, is_bg=1, is_print=1)
    draw_rcnn_targets(img, resampled_rois, target_inds,  targets, cfg, is_before=1, is_after=1, is_print=1)



    im_show('img',img)
    cv2.waitKey(0)

#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 