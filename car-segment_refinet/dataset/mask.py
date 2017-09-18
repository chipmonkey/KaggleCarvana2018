from common import *

from dataset.carvana_cars import *
from dataset.tool import *



#---------------------------------------------------------------------------------
# https://www.kaggle.com/tunguz/baseline-2-optimal-mask
# https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
# submission in "run-length encoding"
# def run_length_encode0(mask):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     inds = np.where(mask.flatten()==1)[0]
#     inds = inds + 1  # one-indexed : https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37108
#     runs = []
#     prev = -1
#     for i in inds:
#         if (i > prev + 1): runs.extend((i, 0)) #They are one-indexed
#         runs[-1] += 1
#         prev = i
#
#     rle = ' '.join([str(r) for r in runs])
#     return rle

#https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    inds[ 0] = 0
    inds[-1] = 0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

def run_length_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H*W),np.uint8)
    runs = np.array([int(s) for s in rle.split(' ')]).reshape(-1,2)
    for r in runs:
        start = r[0]-1       #one-indexed
        end   = start +r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(H,W)
    return mask



#mask weighing

#----------------------------------------------------------------------------------
def run_check_all_rle():

    #check with train_masks.csv given
    csv_file  = CARVANA_DIR + '/masks_train.csv'  # read all annotations
    mask_dir  = CARVANA_DIR + '/annotations/train_gif'  # read all annotations


    df  = pd.read_csv(csv_file)
    for n in range(20): #check 20
        shortname = df.values[n][0].replace('.jpg','')
        rle_hat   = df.values[n][1]

        mask_file = mask_dir + '/' + shortname + '_mask.gif'
        mask_hat = PIL.Image.open(mask_file)
        mask_hat = np.array(mask_hat).astype(np.uint8)

        # check encode
        rle = run_length_encode(mask_hat)
        match = rle == rle_hat
        print('encode @%d : match=%s'%(n,match))

        # check decode
        mask = run_length_decode(rle, H=1280, W=1918, fill_value=1)
        match = np.array_equal(mask, mask_hat)
        print('decode @%d : match=%s'%(n,match))


def run_check_distance_transform():

    mask_file = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/train256x256/0cdf5b5d0ce1_14_mask.png'
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)//255
    invert_mask = 1-mask


    d = cv2.distanceTransform(mask,cv2.DIST_L2,3)
    d = (1-d/d.max())


    invert_d = cv2.distanceTransform(invert_mask,cv2.DIST_L2,3)
    invert_d = (1-invert_d/invert_d.max())

    w = d*mask
    invert_w = invert_d*invert_mask

    im_show('mask', mask*255, resize=1)
    im_show('invert_mask', invert_mask*255, resize=1)
    im_show('d', d*255, resize=1)
    im_show('invert_d', invert_d*255, resize=1)
    im_show('w', w*255, resize=1)
    im_show('invert_w', invert_w*255, resize=1)
    cv2.waitKey(0)


def run_check_averaging():

    img_file = '/media/ssd/data/kaggle-carvana-cars-2017/images/train256x256/0cdf5b5d0ce1_14.jpg'
    image = cv2.imread(img_file)

    mask_file = '/media/ssd/data/kaggle-carvana-cars-2017/annotations/train256x256/0cdf5b5d0ce1_14_mask.png'
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255

    m = torch.from_numpy(mask.reshape(1,1,256,256))
    m = Variable(m).cuda()
    a = F.avg_pool2d(m,kernel_size=11,padding=5,stride=1)
    w = Variable(torch.tensor.torch.ones(256, 256)).cuda()

    ind = a.ge(0.01) * a.le(0.99)
    ind = ind.float()
    w   = ind  #w + ind*1
    w   = w/w.max()

    average = a.data.cpu().numpy()
    average = np.squeeze(average)

    weight = w.data.cpu().numpy()
    weight = np.squeeze(weight)

    # valid = np.where(np.logical_and(average>=0.01, average<=0.99))
    # weight = np.zeros((256,256),np.float32)
    # weight[valid[0],valid[1]] =1

    p = np.zeros((256, 256, 3),np.uint8)
    p[:,:,1] = weight*255
    results = cv2.addWeighted(image, 1, p, 1., 0.)

    print(average)
    print(weight)
    im_show('average', average*255, resize=1)
    im_show('weight', weight*255, resize=1)
    im_show('img', image, resize=1)
    im_show('results', results, resize=1)
    cv2.waitKey(0)
    pass

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_averaging()

    print('\nsucess!')