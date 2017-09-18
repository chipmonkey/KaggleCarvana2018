# find the dice loss limit of resizing
from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py


def run_find_limit():

    img_dir = CARVANA_DIR + '/annotations/train'  # read all annotations
    H,W = 512, 512  #0.997947997061
    H,W = 256, 256  #0.996000875723
    H,W = 128, 128  #0.99203487
    H,W = 418, 627
    H,W = CARVANA_HEIGHT*2, CARVANA_WIDTH*2


    img_list = glob.glob(img_dir + '/*.gif')
    num_imgs = len(img_list)

    all_loss = 0
    for n in range(num_imgs):
        print('n/num_imgs=%d/%d'%(n,num_imgs))

        img_file = img_list[n]
        img = PIL.Image.open(img_file)
        img = np.array(img)

        label = img

        #downsize
        img = cv2.resize(img.astype(np.float32),(W,H), interpolation=cv2.INTER_LINEAR) #INTER_LINEAR #INTER_CUBIC #INTER_AREA #INTER_LANCZOS4
        img = (img>0.5).astype(np.float32)

        # upsize again
        mask = cv2.resize(img,(CARVANA_WIDTH,CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR)
        mask = mask >0.5


        #loss
        l = one_dice_loss_py(mask,label)
        all_loss += l

    all_loss = all_loss/num_imgs
    print(all_loss)
    pass


#make correction
def make_correction():
    img_file='/root/share/data/kaggle-carvana-cars-2017/others/correction/29bb3ece3180_11.jpg'
    save_file='/root/share/data/kaggle-carvana-cars-2017/images/test1024x1024/29bb3ece3180_11.jpg'

    img=cv2.imread(img_file)
    img=cv2.resize(img,(1024,1024))

    im_show('img',img)
    #cv2.waitKey(0)

    cv2.imwrite(save_file,img)


def make_test_bounding_box():

    box_file = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/test_100064.boxes.txt'

    img_dir   = CARVANA_DIR + '/images/test'
    split_file = CARVANA_DIR +'/split/test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip().replace('<replace>/','') for name in names]
    num   = len(names)


    probs_file ='/root/share/project/kaggle-carvana-cars/results/__old_1__/xx3-UNet256_2/submit/probs.npy'
    probs = np.load(probs_file)
    num_probs,H,W = probs.shape

    #fixe corrupted image
    # https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37247
    if 1:
        n0 = names.index('29bb3ece3180_11')
        #mask = cv2.imread('/root/share/project/kaggle-carvana-cars/data/others/ave/11.png',cv2.IMREAD_GRAYSCALE)
        mask = np.zeros((CARVANA_HEIGHT,CARVANA_WIDTH),np.uint8)
        cv2.rectangle(mask,(80*3,75*3),(472*3,305*3),(255,255,255),-1)
        p    = cv2.resize(mask,(H,W)).astype(np.float32)/255
        probs[n0] = p


    masks = probs>0.5
    q = masks.reshape(-1,W)
    h0 = q.sum(axis=1)
    h1 = h0.reshape(-1,H)
    h2 = h1.sum(axis=0)

    q = masks.transpose(0,2,1).reshape(-1,H)
    v0 = q.sum(axis=1)
    v1 = v0.reshape(-1,W)
    v2 = v1.sum(axis=0)

    boxes =  np.zeros((num,4),np.int32)

    #check
    for n in range(0,num):

        h = h1[n]!=0
        idx = np.where(h!=0)[0]
        y0,y1 = idx[0],idx[-1]

        v = v1[n]!=0
        idx = np.where(v!=0)[0]
        x0,x1 = idx[0],idx[-1]


        p  = (probs[n]*255).astype(np.uint8)
        cv2.line(p,(0,y0),(W-1,y0),(128),1)
        cv2.line(p,(0,y1),(W-1,y1),(128),1)
        cv2.line(p,(x0,0),(x0,H-1),(128),1)
        cv2.line(p,(x1,0),(x1,H-1),(128),1)
        draw_shadow_text(p, '256x256 initial prediction', (5,15), 0.5,(255,255,255),1)


        img = cv2.imread(img_dir+ '/' + names[n] + '.jpg' )
        y0 = int(y0/H*CARVANA_HEIGHT)
        y1 = int(y1/H*CARVANA_HEIGHT)
        x0 = int(x0/W*CARVANA_WIDTH)
        x1 = int(x1/W*CARVANA_WIDTH)
        boxes[n]=x0,y0,x1,y1

        cv2.line(img,(0,y0),(CARVANA_WIDTH-1,y0),(0,0,255),10)
        cv2.line(img,(0,y1),(CARVANA_WIDTH-1,y1),(0,0,255),10)
        cv2.line(img,(x0,0),(x0,CARVANA_HEIGHT-1),(0,0,255),10)
        cv2.line(img,(x1,0),(x1,CARVANA_HEIGHT-1),(0,0,255),10)


        im_show('p',p,1)
        im_show('img',img,0.33)
        cv2.waitKey(1)


    #save
    np.savetxt(box_file,boxes,fmt='%d')



def make_train_bounding_box():

    box_file = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/train_5088.boxes.txt'

    img_dir   = CARVANA_DIR + '/images/train'
    split_file = CARVANA_DIR +'/split/train_5088'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip().replace('<replace>/','') for name in names]
    num   = len(names)

    H,W = 512,512
    probs = np.zeros((num,H,W),np.float32)
    for n in range(0,num):
        mask = cv2.imread('/media/ssd/data/kaggle-carvana-cars-2017/annotations/train'+ '/' + names[n] + '_mask.png',cv2.IMREAD_GRAYSCALE)
        p    = cv2.resize(mask,(H,W)).astype(np.float32)/255
        probs[n] = p

    masks = probs>0.5
    q = masks.reshape(-1,W)
    h0 = q.sum(axis=1)
    h1 = h0.reshape(-1,H)
    h2 = h1.sum(axis=0)

    q = masks.transpose(0,2,1).reshape(-1,H)
    v0 = q.sum(axis=1)
    v1 = v0.reshape(-1,W)
    v2 = v1.sum(axis=0)

    boxes =  np.zeros((num,4),np.int32)

    #check
    for n in range(0,num):

        h = h1[n]!=0
        idx = np.where(h!=0)[0]
        y0,y1 = idx[0],idx[-1]

        v = v1[n]!=0
        idx = np.where(v!=0)[0]
        x0,x1 = idx[0],idx[-1]


        p  = (probs[n]*255).astype(np.uint8)
        cv2.line(p,(0,y0),(W-1,y0),(128),1)
        cv2.line(p,(0,y1),(W-1,y1),(128),1)
        cv2.line(p,(x0,0),(x0,H-1),(128),1)
        cv2.line(p,(x1,0),(x1,H-1),(128),1)
        draw_shadow_text(p, '256x256 initial prediction', (5,15), 0.5,(255,255,255),1)


        img = cv2.imread(img_dir+ '/' + names[n] + '.jpg' )
        y0 = int(y0/H*CARVANA_HEIGHT)
        y1 = int(y1/H*CARVANA_HEIGHT)
        x0 = int(x0/W*CARVANA_WIDTH)
        x1 = int(x1/W*CARVANA_WIDTH)
        boxes[n]=x0,y0,x1,y1

        cv2.line(img,(0,y0),(CARVANA_WIDTH-1,y0),(0,0,255),10)
        cv2.line(img,(0,y1),(CARVANA_WIDTH-1,y1),(0,0,255),10)
        cv2.line(img,(x0,0),(x0,CARVANA_HEIGHT-1),(0,0,255),10)
        cv2.line(img,(x1,0),(x1,CARVANA_HEIGHT-1),(0,0,255),10)


        im_show('p',p,1)
        im_show('img',img,0.33)
        cv2.waitKey(1)


    #save
    np.savetxt(box_file,boxes,fmt='%d')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_train()
    #make_correction()
    #make_test_bounding_box()
    make_train_bounding_box()

    print('\nsucess!')