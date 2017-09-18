# reference:  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py

from dataset.annotation import *
from dataset.box import *


# rpn target
def rpn_compute_targets(et_boxes, gt_boxes):
    ''' Compute bounding-box regression targets for an image.'''

    assert et_boxes.shape[0] == gt_boxes.shape[0]
    assert et_boxes.shape[1] == 4
    assert gt_boxes.shape[1] == 4
    targets = box_transform(et_boxes, gt_boxes).astype(np.float32, copy=False)

    return targets

# for debug (draw results of ground truth generation)-------------------------
def draw_bases(img, bases ):

    height, width = img.shape[0:2]
    cx,cy = width//2,height//2
    num =len(bases)
    for n in range(num):
        b = bases[n]
        draw_dotted_rect(img,(b[0]+cx,b[1]+cy),(b[2]+cx,b[3]+cy),(0,255,255),1,gap=3)

    cv2.circle(img,(cx,cy),2, (0,255,255), -1, cv2.LINE_AA)


def draw_rpn_labels(img, windows, label_inds, labels, is_fg=1, is_bg=1, is_print=1):
    label_inds = label_inds.cpu().data.numpy()
    labels     = labels.cpu().data.numpy()

    ## red  + dot : +ve label
    ## grey + dot : -ve label

    ## draw +ve/-ve labels ......
    num_windows = len(windows)
    labels = labels.reshape(-1)

    fg_label_inds = label_inds[np.where(labels == 1)[0]]
    bg_label_inds = label_inds[np.where(labels == 0)[0]]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print:
        print ('rpn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    if is_bg:
        for i in bg_label_inds:
            a = windows[i]
            cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (32,32,32), 1)
            cv2.circle(img,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (32,32,32), -1, cv2.LINE_AA)
    if is_fg:
        for i in fg_label_inds:
            a = windows[i]
            cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
            cv2.circle(img,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (0,0,255), -1, cv2.LINE_AA)



def draw_rpn_targets(img, windows, target_inds, targets, is_before=1, is_after=1, is_print=1):
    target_inds = target_inds.cpu().data.numpy()
    targets     = targets.cpu().data.numpy()

    #draw +ve targets ......
    num_pos_target = len(target_inds)
    if is_print:
        print ('rpn target : num_pos=%d'  %(num_pos_target))


    for n,i in enumerate(target_inds):
        a = windows[i]
        t = targets[n]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        if is_before:
            cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
            cv2.circle(img,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (0,0,255), -1, cv2.LINE_AA)

        if is_after:
            cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), (0,255,255), 1)



# Faster-rcnn ground-truth layer rpn----------------------------------------

def rpn_target(windows, inside_inds, annotation, cfg):

    gt_boxes = annotation['boxes']

    # label: 1 is positive, 0 is negative, -1 is dont care
    inside_windows     = windows[inside_inds, :]
    num_inside_windows = len(inside_windows)
    num_gt_boxes       = len(gt_boxes)

    inside_labels = np.empty(num_inside_windows, dtype=np.float32)
    inside_labels.fill(-1)

    # overlaps between the anchors and the gt process
    overlaps = box_overlap(
        np.ascontiguousarray(inside_windows, dtype=np.float),
        np.ascontiguousarray(gt_boxes,       dtype=np.float)
    )

    argmax_overlaps    = overlaps.argmax(axis=1)
    max_overlaps       = overlaps[np.arange(num_inside_windows), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps    = overlaps[gt_argmax_overlaps, np.arange(num_gt_boxes)]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]   # include multiple maxs

    inside_labels[max_overlaps <  cfg.rpn.train_bg_thresh_high] = 0  # bg label
    inside_labels[max_overlaps >= cfg.rpn.train_fg_thresh_low ] = 1  # fg label: above threshold IOU
    inside_labels[gt_argmax_overlaps] = 1                            # fg label: for each gt, window with highest overlap


    # subsample positive labels
    num_fg = int(cfg.rpn.train_fg_fraction * cfg.rpn.train_batch_size)
    fg_inds = np.where(inside_labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice( fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        inside_labels[disable_inds] = -1

    # subsample negative labels
    num_bg  = cfg.rpn.train_batch_size - np.sum(inside_labels == 1)
    bg_inds = np.where(inside_labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        inside_labels[disable_inds] = -1

    # training windows for classification labels
    idx        = np.where(inside_labels != -1)[0]
    label_inds = inside_inds[idx]
    labels     = inside_labels[idx]

    # training windows for regression targets
    idx          = np.where(inside_labels ==  1)[0]
    target_inds  = inside_inds[idx]
    pos_windows  = inside_windows[idx]
    pos_gt_boxes = gt_boxes[argmax_overlaps[idx]]
    targets = box_transform(pos_windows, pos_gt_boxes)

    label_inds  = Variable(torch.from_numpy(label_inds).type(torch.cuda.LongTensor))
    labels      = Variable(torch.from_numpy(labels).type(torch.cuda.LongTensor))
    target_inds = Variable(torch.from_numpy(target_inds).type(torch.cuda.LongTensor))
    targets     = Variable(torch.from_numpy(targets).type(torch.cuda.FloatTensor))

    return label_inds, labels, target_inds, targets






def check_layer():
    annotation_file = '/media/ssd/[data]/dummy/object-det-debug/pencil/annotations/000.txt'
    img_file        = '/media/ssd/[data]/dummy/object-det-debug/pencil/images/000.jpg'
    #annotation_file = '/media/ssd/[data]/dummy/object-det-debug/bird_and_flower/annotations/000.txt'
    #img_file        = '/media/ssd/[data]/dummy/object-det-debug/bird_and_flower/images/000.jpg'
    # annotation_file = '/media/ssd/[data]/dummy/object-det-debug/rose/annotations/000.txt'
    # img_file        = '/media/ssd/[data]/dummy/object-det-debug/rose/images/000.jpg'

    # set some dummy data
    annotation = load_annotation(annotation_file, img_file)
    img        = cv2.imread(img_file)


    # check layer
    cfg = Configuration() #default configuration
    cfg.rpn.train_fg_thresh_low = 0.5
    cfg.rpn.scales=[128,256]  #128
    cfg.rpn.ratios=[0.1]

    height, width  = img.shape[0:2]
    allowed_border = cfg.rpn.allowed_border
    stride         = cfg.rpn.stride
    image_shape    = (height,width)                     #original image width
    feature_shape  = (height//stride,width//stride)     #feature map width


    bases = make_bases(ratios = np.array(cfg.rpn.ratios), scales = np.array(cfg.rpn.scales))
    windows, inside_inds = make_windows(bases, stride, image_shape, feature_shape, allowed_border)

    label_inds, labels, target_inds, targets = rpn_target(windows, inside_inds, annotation, cfg)

    #check
    gt_boxes = annotation['boxes']

    #draw_bases(img, bases)
    draw_gt_boxes(img, gt_boxes)
    #draw_rpn_labels(img, windows, label_inds, labels, is_fg=1, is_bg=0, is_print=1)
    draw_rpn_targets(img, windows, target_inds, targets)

    im_show('img',img)
    cv2.waitKey(0)

#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 