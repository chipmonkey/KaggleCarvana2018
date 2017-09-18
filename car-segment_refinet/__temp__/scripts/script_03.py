# find the dice loss limit of resizing
#import sys
#sys.path.append('/share/project/kaggle-carvana-cars/build/car-segment_single')
#print(sys.path)

from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py


def align_train_images_by_boxes():


    H,W =  1024,2048

    # small_dir  = CARVANA_DIR + '/images/train%dx%d_align'%(H,W)
    # img_dir         = CARVANA_DIR + '/images/train'
    # train_box_file  = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/train_5088.boxes.txt'
    # train_name_file = '/media/ssd/data/kaggle-carvana-cars-2017/split/train_5088'

    small_dir  = CARVANA_DIR + '/images/test%dx%d_align'%(H,W)
    img_dir         = CARVANA_DIR + '/images/test'
    train_box_file  = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/test_100064.boxes.txt'
    train_name_file = '/media/ssd/data/kaggle-carvana-cars-2017/split/test_100064'


    os.makedirs(small_dir,exist_ok=True)
    boxes = np.loadtxt(train_box_file).astype(np.int32)
    with open(train_name_file) as f:
        names = f.readlines()
    names = [name.strip().replace('<replace>/','') for name in names]

    #check boxes
    hs = boxes[:,3]-boxes[:,1]
    print (max(hs))

    #canoical
    #CARVANA_HEIGHT = 1280
    #CARVANA_WIDTH  = 1918
    #standard_img = np.zeros((1024,1918,3),np.uint8)
    sy0 = int(      0.15*H)
    sy1 = int(1024 -0.15*H)


    num = len(boxes)
    for n in range(num):
        if n%1000==0: print('%d/%d'%(n,num))
        img_file = img_dir+ '/' + names[n] + '.jpg'
        img  = cv2.imread( img_file )
        x0,y0,x1,y1 = boxes[n]
        h = y1-y0
        w = x1-x0

        xx0 = int(x0 - 0.2*w)
        xx1 = int(x1 + 0.2*w)
        yy0 = int(y0 - 0.2*h)
        yy1 = int(y1 + 0.2*h)

        box0 = np.array([ [xx0,yy0], [xx1,yy0],  [xx1,yy1], [xx0,yy1], ]).astype(np.float32)
        box1 = np.array([ [0,0], [W,0],  [W,H], [0,H], ]).astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        simg = cv2.warpPerspective(img, mat, (W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,))
        #cv2.line(simg,(0,sy0),(CARVANA_WIDTH-1,sy0),(0,0,255),10)
        #cv2.line(simg,(0,sy1),(CARVANA_WIDTH-1,sy1),(0,0,255),10)


        save_file = img_file.replace(img_dir, small_dir)
        cv2.imwrite(save_file,simg)

        #im_show('simg',simg,0.33)
        #cv2.waitKey(1)


def align_train_masks_by_boxes():


    H,W =  1024,2048
    small_dir  = CARVANA_DIR + '/annotations/train%dx%d_align'%(H,W)
    os.makedirs(small_dir,exist_ok=True)

    img_dir         = CARVANA_DIR + '/annotations/train'
    train_box_file  = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/train_5088.boxes.txt'
    train_name_file = '/media/ssd/data/kaggle-carvana-cars-2017/split/train_5088'

    boxes = np.loadtxt(train_box_file).astype(np.int32)
    with open(train_name_file) as f:
        names = f.readlines()
    names = [name.strip().replace('<replace>/','') for name in names]

    #check boxes
    hs = boxes[:,3]-boxes[:,1]
    print (max(hs))

    #canoical
    #CARVANA_HEIGHT = 1280
    #CARVANA_WIDTH  = 1918
    #standard_img = np.zeros((1024,1918,3),np.uint8)
    sy0 = int(      0.15*H)
    sy1 = int(1024 -0.15*H)


    num = len(boxes)
    for n in range(num):
        img_file = img_dir+ '/' + names[n] + '_mask.png'
        img  = cv2.imread( img_file )
        x0,y0,x1,y1 = boxes[n]
        h = y1-y0
        w = x1-x0

        xx0 = int(x0 - 0.2*w)
        xx1 = int(x1 + 0.2*w)
        yy0 = int(y0 - 0.2*h)
        yy1 = int(y1 + 0.2*h)

        box0 = np.array([ [xx0,yy0], [xx1,yy0],  [xx1,yy1], [xx0,yy1], ]).astype(np.float32)
        box1 = np.array([ [0,0], [W,0],  [W,H], [0,H], ]).astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        simg = cv2.warpPerspective(img, mat, (W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,))

        save_file = img_file.replace(img_dir, small_dir)
        cv2.imwrite(save_file,simg)

        #cv2.line(simg,(0,sy0),(CARVANA_WIDTH-1,sy0),(0,0,255),10)
        #cv2.line(simg,(0,sy1),(CARVANA_WIDTH-1,sy1),(0,0,255),10)
        #im_show('simg',simg,0.33)
        #cv2.waitKey(1)

def check_upsample():

    batch_size  = 8
    C,H,W = 1,6,8

    data = np.zeros((batch_size,C,H,W),np.float32)
    for n in range(batch_size):
        for c in range(C):
            for y in range(H):
                for x in range(W):
                     data[n,c,y,x]=n*1000+c*100+y*10 +x

    x = Variable(torch.from_numpy(data)).cuda()
    y = F.upsample_bilinear(x,size=(3,5))
    xx=0



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    align_train_images_by_boxes()

    print('\nsucess!')