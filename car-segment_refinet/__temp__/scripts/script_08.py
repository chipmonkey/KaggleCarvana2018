
from common import *
from dataset.carvana_cars import *
from net.tool import *

# visualise car error

# draw -----------------------------------
def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def draw_contour(image, mask, color=(0,255,0), thickness=1):
    threshold = 127
    ret, thresh = cv2.threshold(mask,threshold,255,0)
    ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = ret[0]
    contours  = ret[1]
    #image[...]=image
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)
    ## drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None): # real signature unknown; restored from __doc__


def draw_mask(img,  mask,  color=(0,255,0), alpha=1., beta=0.8):
    mask = np.dstack((mask,mask,mask))*np.array(color)
    mask = mask.astype(np.uint8)
    img[...] = cv2.addWeighted(img,  alpha, mask, beta,  0.) # image * α + mask * β + λ




#debug and show -----------------------------------------------------------------
def draw_contour_on_image(image, label, prob=None):
    results = image.copy()
    if prob is not None:
        draw_contour(results, prob, color=(0,255,0), thickness=1)
    if label is not None:
        draw_contour(results, label, color=(0,0,255), thickness=1)

    return results


def draw_dice_on_image(image, label, prob, name=''):

    label = label>127
    prob  = prob>127
    score = one_dice_loss_py(label, prob)

    H,W,C = image.shape
    results = np.zeros((H*W,3),np.uint8)
    a = (2*label+prob).reshape(-1)
    miss = np.where(a==2)[0]
    hit  = np.where(a==3)[0]
    fp   = np.where(a==1)[0]
    label_sum = label.sum()
    prob_sum  = prob.sum()

    results[miss] = np.array([0,0,255])
    results[hit]  = np.array([64,64,64])
    results[fp]   = np.array([0,255,0])
    results = results.reshape(H,W,3)
    L=30
    draw_shadow_text  (results, '%s'%(name), (5,1*L),  1, (255,255,255), 2)
    draw_shadow_text  (results, '%0.5f'%(score), (5,2*L),  1, (255,255,255), 2)
    draw_shadow_text  (results, 'label = %0.0f'%(label_sum), (5,3*L),  1, (255,255,255), 2)
    draw_shadow_text  (results, 'prob = %0.0f (%0.4f)'%(prob_sum,prob_sum/label_sum), (5,4*L),  1, (255,255,255), 2)
    draw_shadow_text  (results, 'miss = %0.0f (%0.4f)'%(len(miss), len(miss)/label_sum), (5,5*L),  1, (0,0,255), 2)
    draw_shadow_text  (results, 'fp   = %0.0f (%0.4f)'%(len(fp), len(fp)/prob_sum), (5,6*L),  1, (0,255,0), 2)

    return results


def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score




# main #################################################################
CARVANA_HEIGHT = 1280
CARVANA_WIDTH  = 1918

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #example to make visualisation results
    name = '791c1a9775be_06'

    prob_file  = '/root/share/project/kaggle-carvana-cars/deliver/clean_data/code/example_data/prob_1024x1024_%s.npy'%name  #prediction from CNN
    label_file = CARVANA_DIR + '/annotations/%s/%s_mask.png'%('train',name)
    img_file   = CARVANA_DIR + '/images/%s/%s.jpg'%('train',name)

    image = cv2.imread(img_file)
    label = cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)
    prob  = np.load(prob_file)
    prob = cv2.resize(prob,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT))

    #draw hit,miss
    res = draw_dice_on_image(image, label, prob, name)
    cv2.imwrite('/root/share/project/kaggle-carvana-cars/deliver/clean_data/code/example_data/dice.png', res)
    im_show('dice', res, resize=0.33)
    cv2.waitKey(1)


    #draw contour
    res = draw_contour_on_image(image, label, prob)
    cv2.imwrite('/root/share/project/kaggle-carvana-cars/deliver/clean_data/code/example_data/contours.png', res)
    im_show('contour', res, resize=0.33)
    cv2.waitKey(0)


#test on some images
def run_test_images():

    out_dir    = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-weight-00d'
    model_file = out_dir +'/snap/071.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/xxx.code.zip')

    log = Logger()
    log.open(out_dir+'/log.xxx.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 8

    test_dataset = KgCarDataset( 'new',  'new',#100064  #3197
                                 #'valid_v0_768',  'train1024x1024',#100064  #3197
                                     transform= [
                                        lambda x:  fix_resize(x,512,512),
                                    ],mode='test')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)

    log.write('\tbatch_size         = %d\n'%batch_size)
    log.write('\ttest_dataset.split = %s\n'%test_dataset.split)
    log.write('\tlen(test_dataset)  = %d\n'%len(test_dataset))
    log.write('\n')


    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    ## start testing now #####
    log.write('start prediction ...\n')
    start = timer()

    net.eval()


    for it, (images, indices) in enumerate(test_loader, 0):

        images = Variable(images,volatile=True).cuda()
        logits = net(images)
        probs  = F.sigmoid(logits)

        show_batch_results(indices, images, probs, labels=None, wait=0,
                out_dir=out_dir+'/xxx', mode='by_name', names=test_dataset.names, df=None, epoch=0, it=0)

    #----------------------------------


