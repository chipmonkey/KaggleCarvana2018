
from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py
from net.segmentation.loss import *

# test pytorch upsample error


def try_pytorch_upsample_error():


    H,W = CARVANA_HEIGHT, CARVANA_WIDTH
    H_small,W_small = 256,256  #H//2,W//2

    mask_dir = CARVANA_DIR + '/annotations/train'  # read all annotations
    img_list = sorted(glob.glob(mask_dir + '/*.png'))
    num_imgs = len(img_list)

    batch_size = 2
    acc = 0
    sum = 0
    for n in range(0, num_imgs, batch_size):

        B = batch_size if n+batch_size <= num_imgs else num_imgs-n
        print('n/num_imgs=%05d/%05d,  B = %d'%(n,num_imgs,B))

        labels = np.zeros((B,1,H,W),np.float32)
        for b in range(B):
            m = n+b
            mask_file = img_list[m]
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            label = mask/255
            labels[b] = label

            #debug
            #print (mask_file)
            #im_show('label', label*255, resize=0.25)
            #cv2.waitKey(1)

        labels = Variable(torch.from_numpy(labels),volatile=True).cuda()

        #downsize + upsize
        labels_small = F.upsample(labels, size=(H_small,W_small),mode='bilinear')
        labels_small = (labels_small>0.5).float()
        predicts     = F.upsample(labels_small, size=(H,W),mode='bilinear')
        predicts     = (predicts>0.5).float()

        a = DiceAccuracy()(predicts,labels)
        acc += a.data[0]*B
        sum += B


    acc = acc/sum
    print(sum)
    print(acc)  #0.9997714872255266 for half resolution
                #0.9965334242244936 for 256,256
                #0.99600            for 256,256(threshold), ref : opencv 0.996
                #0.9986002906313483 for 640,959(threshold)
                #0.9986002906313483 for 640,960(threshold)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    try_pytorch_upsample_error()

    print('\nsucess!')