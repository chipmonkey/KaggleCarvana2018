from common import *
from submit import *
from dataset.carvana_cars import *

from net.tool import *
from net.rate import *
from net.segmentation.loss import *
from post_processing import *


#from net.segmentation.my_unet import UNet1024 as Net
#from net.segmentation.my_unet import MoreUNet as Net
#from net.segmentation.my_frrunet import FRRUNet512 as Net


#from net.segmentation.my_unet_baseline import UNet256 as Net
#from net.segmentation.my_unet import PriorUNet128 as Net

## ----------------------------------------------------------------------------------
#  ffmpeg -y -loglevel 0 -f image2 -r 15 -i 'rcnn_post_nms/%*.jpg' -b:v 8000k rcnn_post_nms.avi
#  ffmpeg -i results.avi -vf scale=300:100 -b:v 8000k results-small.avi



## experiment setting here ----------------------------------------------------
def criterion(logits, labels, is_weight=True):
    #l = BCELoss2d()(logits, labels)
    #l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)

    #compute weights
    a   = F.avg_pool2d(labels,kernel_size=11,padding=5,stride=1)
    ind = a.ge(0.01) * a.le(0.99)
    ind = ind.float()
    weights  = Variable(torch.tensor.torch.ones(a.size())).cuda()

    if is_weight:
        w0 = weights.sum()
        weights = weights + ind*2
        w1 = weights.sum()
        weights = weights/w1*w0

    l = WeightedBCELoss2d()(logits, labels, weights) + \
        WeightedSoftDiceLoss()(logits, labels, weights)

    return l


#https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
#https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score

#https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2, is_average =True):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    if is_average:
        score = scores.sum()/num
        return score
    else:
        return scores



#debug and show ----------------------------------------------------------
def show_batch_results(indices, images, probs, labels=None, wait=1,
                       out_dir=None, names=None, mode='both', epoch=0, it=0):

    images = (images.data.cpu().numpy()*255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))
    probs  = (probs.data.cpu().numpy()*255).astype(np.uint8)
    if labels is not None:
        labels = (labels.data.cpu().numpy()*255).astype(np.uint8)

    batch_size,H,W,C = images.shape
    for b in range(batch_size):
        image = images[b]
        prob  = probs[b]
        label = None
        score = -1
        if labels is not None:
            label = labels[b]
            score = one_dice_loss_py(label>127 , prob>127)

        results = make_combined_image(image, label=label, prob=prob)
        name = names[indices[b]]
        draw_shadow_text  (results, '%s.jpg'%(name), (5,15),  0.5, (255,255,255), 1)
        draw_shadow_text  (results, '%0.5f'%(score), (5,30),  0.5, (255,255,255), 1)


        if out_dir is not None:
            os.makedirs(out_dir+'/results_by_name', exist_ok=True)
            os.makedirs(out_dir+'/results_by_score', exist_ok=True)
            #cv2.imwrite(save_dir + '/%05d-%03d.jpg'%(it,b), results)
            if mode in['both','by_score'] : cv2.imwrite(out_dir+'/results_by_score/%0.5f-%s.jpg'%(score,name), results)
            if mode in['both','by_name' ] : cv2.imwrite(out_dir+'/results_by_name/%s.jpg'%(name), results)

        im_show('train',  results,  resize=1)
        cv2.waitKey(wait)





#helper ----------------------------------------------------------
# def predict8(net, test_loader):
#
#     test_dataset = test_loader.dataset
#
#     num = len(test_dataset)
#     H, W = CARVANA_IMAGE_H, CARVANA_IMAGE_W
#     predictions  = np.zeros((num, H, W),np.uint8)
#
#     test_num  = 0
#     for it, (images, indices) in enumerate(test_loader, 0):
#         batch_size = len(indices)
#         test_num  += batch_size
#
#         # forward
#         images = Variable(images.cuda(),volatile=True)
#         logits = net(images)
#         probs  = F.sigmoid(logits)
#
#         probs = probs.data.cpu().numpy().reshape(-1, H, W)
#         predictions[test_num-batch_size : test_num] = probs*255
#
#     assert(test_num == len(test_loader.sampler))
#     return predictions

CSV_BLOCK_SIZE = 32000
#  100064/32000 =
def predict8_in_blocks(net, test_loader, block_size=CSV_BLOCK_SIZE, log=None, save_dir=None):

    test_dataset = test_loader.dataset
    test_iter    = iter(test_loader)
    test_num     = len(test_dataset)
    batch_size   = test_loader.batch_size
    assert(block_size%batch_size==0)
    assert(log!=None)

    start0 = timer()
    num  = 0
    predictions = []
    for n in range(0, test_num, block_size):
        M = block_size if n+block_size < test_num else test_num-n
        log.write('[n=%d, M=%d]  \n'%(n,M) )

        start = timer()

        p = None #np.zeros((M, H, W),np.uint8)
        for m in range(0, M, batch_size):
            print('\r\t%05d/%05d'%(m,M), end='',flush=True)

            images, indices  = test_iter.next()
            if images is None:
                break


            # forward
            images = Variable(images.cuda(),volatile=True)
            logits = net(images)
            probs  = F.sigmoid(logits)

            # save results
            if p is None:
                H = images.size(2)
                W = images.size(3)
                p = np.zeros((M, H, W),np.uint8)
                ids = np.zeros((M),np.int64)

            batch_size = len(indices)
            probs = probs.data.cpu().numpy().reshape(-1, H, W)
            probs = probs*255
            p[m : m+batch_size] = probs
            ids[m : m+batch_size] = indices.cpu().numpy()
            num  += batch_size
            #print('\tm=%d, m+batch_size=%d'%(m,m+batch_size) )
            #show_test_results(test_loader.dataset, probs, indices, wait=0, save_dir=None)

        pass # end of one block -----------------------------------------------------
        print('\r')
        log.write('\tpredict = %0.2f min, '%((timer() - start) / 60))
        predictions.append(p)

        if save_dir is not None:
            start = timer()
            j = len(predictions)-1
            np.savetxt(save_dir+'/indices-part%02d.8.txt'%(j), ids, fmt='%d')
            np.save(save_dir+'/probs-part%02d.8.npy'%(j), p) #  1min
            log.write('save = %0.2f min'%((timer() - start) / 60))
        log.write('\n')


    log.write('\nall time = %0.2f min\n'%((timer() - start0) / 60))
    assert(test_num == num)
    return predictions



def evaluate(net, test_loader):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    test_acc  = 0
    test_loss = 0
    test_num  = 0
    for it, (images, priors, labels, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)
        priors = Variable(labels.cuda(),volatile=True)
        labels = Variable(labels.cuda(),volatile=True)

        # forward
        logits = net(images, priors)
        probs  = F.sigmoid(logits)
        masks  = (probs>0.5).float()

        loss = criterion(logits, labels)
        acc  = dice_loss(masks, labels)

        batch_size = len(indices)
        test_num  += batch_size
        test_loss += batch_size*loss.data[0]
        test_acc  += batch_size*acc.data[0]

    assert(test_num == len(test_loader.sampler))
    test_loss = test_loss/test_num
    test_acc  = test_acc/test_num

    return test_loss, test_acc





def show_test_results(test_dataset, probs, indices=None, N=100, wait=1, save_dir=None):

        num_test = len(probs)
        for i in range(num_test):
            if indices is not None:
                n =  indices[i]
            else:
                n = i  #n = random.randint(0,num_test-1) #n = i

            shortname    = test_dataset.names[n].split('/')[-1].replace('.jpg','')
            image, index = test_dataset[n]
            image        = tensor_to_image(image, std=255)
            prob         = probs[i]
            results = make_combined_image(image, label=None, prob=prob)

            if save_dir is not None:
                cv2.imwrite(save_dir + '/%s.jpg'%shortname, results)

            print(shortname)
            im_show('test',  results,  resize=0.33)
            cv2.waitKey(wait)




# ------------------------------------------------------------------------------------
def run_train():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet256-weight-id2-labels-prior-03'
    initial_checkpoint = None #'/root/share/project/kaggle-carvana-cars/results/single/UNet128-weight-id2-labels-prior-02/checkpoint/008.pth'
        #None #'/root/share/project/kaggle-carvana-cars/results/single/UNet1024_crop00_01b/checkpoint/015.pth'

    #logging, etc --------------------
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir+'/train/results', exist_ok=True)
    os.makedirs(out_dir+'/valid/results', exist_ok=True)
    os.makedirs(out_dir+'/test/results',  exist_ok=True)
    os.makedirs(out_dir+'/backup', exist_ok=True)
    os.makedirs(out_dir+'/checkpoint', exist_ok=True)
    os.makedirs(out_dir+'/snap', exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/train.code.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** experiment for average labels channel as prior**\n\n')
    
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')




    ## dataset ----------------------------------------
    def train_augment(image, prior, label):
        image, prior, label = random_horizontal_flipN([image, prior, label])
        image, prior, label = random_shift_scale_rotateN([image, prior, label], shift_limit=(-0.0625,0.0625),
                  scale_limit=(-0.09,0.121), rotate_limit=(-0,0))

        #image, label = random_mask_hue(image, label, hue_limit=(-1,1), u=0.5)
        #image = random_hue(image, hue_limit=(-1,1), u=0.5)
        image = random_brightness(image, limit=(-0.5,0.5), u=0.5)
        image = random_contrast  (image, limit=(-0.5,0.5), u=0.5)
        image = random_saturation(image, limit=(-0.3,0.3), u=0.5)
        image = random_gray(image, u=0.25)

        return  image, prior, label

    log.write('** dataset setting **\n')
    batch_size   =  16
    num_grad_acc =  16//batch_size


    train_dataset = KgCarPriorDataset(  'train_v0_4320', 'train256x256', ## 'train_160', ##128x128 ##256x256
                                transform = [ lambda x,prior,y:train_augment(x,prior,y), ], mode='train')
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True)
    ##check_dataset(train_dataset, train_loader), exit(0)

    valid_dataset = KgCarPriorDataset('valid_v0_768', 'train256x256',
                                transform=[], mode='train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 6,
                        pin_memory  = True)


    test_dataset = KgCarPriorDataset( 'test_3197', 'test',
                                transform=[], mode='test')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 5,
                        pin_memory  = True)


    log.write('\ttrain_dataset.split = %s\n'%train_dataset.split)
    log.write('\tvalid_dataset.split = %s\n'%valid_dataset.split)
    log.write('\ttest_dataset.split  = %s\n'%test_dataset.split)
    log.write('\tlen(train_dataset)  = %d\n'%len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n'%len(valid_dataset))
    log.write('\tlen(test_dataset)   = %d\n'%len(test_dataset))
    log.write('\n%s\n\n'%(inspect.getsource(train_augment)))


    ## net ----------------------------------------
    log.write('** net setting **\n')

    net = Net(in_shape=(4, 128, 128))
    net.cuda()
    log.write('%s\n\n'%(type(net)))
    log.write('%s\n\n'%(str(net)))
    log.write('%s\n\n'%(inspect.getsource(net.__init__)))
    log.write('%s\n\n'%(inspect.getsource(net.forward )))

    ## optimiser ----------------------------------
    num_epoches = 50  #100
    it_print    = 1   #20
    it_smooth   = 20
    epoch_valid = 1
    epoch_test  = 5
    epoch_save  = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,24, 26, 28, 30, 31, 32, 33, 34, 35, 40, 45, 50, num_epoches-1]
    LR = StepLR([ (0, 0.01),  (35, 0.005),  (40,0.001),  (42, 0.0001),(44, -1)]) #bn mode
    #LR = StepLR([ (0, 0.01),  (10, 0.005),  (40,0.001),  (42, 0.0001),(44, -1)]) #bn mode

    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9, weight_decay=0.0005)
    #optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)


    ## resume from previous ------------------------
    start_epoch=0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    #training ####################################################################3
    log.write('** start training here! **\n')
    log.write(' num_grad_acc x batch_size = %d x %d=%d\n'%(num_grad_acc,batch_size,num_grad_acc*batch_size))
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' LR=%s\n\n'%str(LR) )
    log.write('\n')


    log.write('epoch    iter      rate   | valid_loss/acc | train_loss/acc | batch_loss/acc ... \n')
    log.write('--------------------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    smooth_acc  = 0.0
    train_loss = np.nan
    train_acc  = np.nan
    valid_loss = np.nan
    valid_acc  = np.nan
    time = 0
    start0 = timer()


    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        start = timer()

        #---learning rate schduler ------------------------------
        lr = LR.get_rate(epoch, num_epoches)
        if lr<0 :
            break
        adjust_learning_rate(optimizer, lr/num_grad_acc)

        # if epoch>=28:
        #     adjust_learning_rate(optimizer, lr=0.005)
        # if epoch>=num_epoches-2:
        #     adjust_learning_rate(optimizer, lr=0.001)


        rate = get_learning_rate(optimizer)[0]*num_grad_acc #check
        #--------------------------------------------------------


        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0
        num_its = len(train_loader)

        net.train()
        for it, (images, priors, labels, indices) in enumerate(train_loader, 0):
            images  = Variable(images).cuda()
            priors  = Variable(priors).cuda()
            labels  = Variable(labels).cuda()

            #forward
            logits = net(images, priors)
            probs  = F.sigmoid(logits)
            masks  = (probs>0.5).float()


            loss = criterion(logits, labels)
            acc  = dice_loss(masks, labels)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # accumulate gradients
            if it==0:
                optimizer.zero_grad()
            loss.backward()
            if it%num_grad_acc==0:
                optimizer.step()
                optimizer.zero_grad()  # assume no effects on bn for accumulating grad


            # print statistics
            batch_acc  = acc.data [0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if it%it_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.0
                sum_train_acc  = 0.0
                sum = 0

            if it%it_print == 0 or it==num_its-1:
                print('\r%5.1f   %5d    %0.4f   | ......  ...... | %0.4f  %0.4f | %0.4f  %0.4f ' % \
                        (epoch + (it+1)/num_its, it+1, rate, train_loss, train_acc, batch_loss, batch_acc),\
                        end='',flush=True)

            #debug show prediction results ---
            if 0:
            #if it%100==0:
                show_batch_results(indices, images, probs, labels,
                                   wait=1, save_dir=out_dir+'/train/results', names=train_dataset.names, epoch=epoch, it=it)

        end  = timer()
        time = (end - start)/60
        #end of epoch --------------------------------------------------------------


        if epoch % epoch_valid == 0 or epoch == 0 or epoch == num_epoches-1:
            net.eval()
            valid_loss, valid_acc = evaluate(net, valid_loader)

            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, time))

        # if 0:
        # #if epoch % epoch_test == 0 or epoch == 0 or epoch == num_epoches-1:
        #     net.eval()
        #     probs = predict(net, test_loader)
        #     show_test_results(test_dataset, probs, wait=1, save_dir=out_dir+'/test/results/')


        if 1:
        #if epoch in epoch_save:
            torch.save(net.state_dict(),out_dir +'/snap/%03d.pth'%epoch)
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/checkpoint/%03d.pth'%epoch)
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py


    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60
    log.write('\nalltime = %f min\n'%time0)

    ## check : load model and re-test
    torch.save(net.state_dict(),out_dir +'/snap/final.pth')

# ------------------------------------------------------------------------------------

def run_make_validate_results():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet128-weight-norm-00'
    #out_dir  = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-00'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/valid/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/make_validate.code.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    def valid_augment(image,label):
        #image, label = random_horizontal_flip2(image, label,u=1)
        # image, label = random_shift_scale_rotate2(image, label, shift_limit=(-0.0625,0.0625),
        #           scale_limit=(-0.09,0.121), rotate_limit=(-0,0))

        #image, label = random_hue2(image, label, hue_limit=(-1,1), u=0.5)
        #image = random_hue(image, hue_limit=(-0.5,-0.5), u=1)
        #image = random_brightness(image, limit=(0.5,0.5), u=1)
        #image = random_contrast  (image, limit=(-0.5,-0.5), u=1)
        # image = random_saturation(image, limit=0.3, u=0.5)
        #image = random_gray(image, u=1)

        return  image, label





    batch_size   =  16
    valid_dataset = KgCarTrainDataset('valid_v0_768', 'train128x128',
                                transform=[
                                    lambda x,y:  valid_augment(x, y),
                                ])
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 6,
                        pin_memory  = True)
    ##check_dataset(valid_dataset, valid_loader), exit(0)

    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    num_valid = len(valid_dataset)
    names = valid_dataset.names
    all_accs   =np.zeros(num_valid,np.float32)
    all_indices=np.zeros(num_valid,np.float32)
    net.train()

    start=0
    end  =0
    for it, (images, labels, indices) in enumerate(valid_loader, 0):
        images  = Variable(images).cuda()
        labels  = Variable(labels).cuda()
        batch_size = len(indices)

        #forward
        logits = net(images)
        probs  = F.sigmoid(logits)
        masks  = (probs>0.5).float()

        loss = criterion(logits, labels)
        accs = dice_loss(masks, labels, is_average=False)
        acc  = accs.sum()/batch_size


        end = start + batch_size
        all_accs   [start:end]=accs.data.cpu().numpy()
        all_indices[start:end]=indices.cpu().numpy()
        start=end

        show_batch_results(indices, images, probs, labels=labels, wait=1,
                           out_dir=out_dir+'/valid', mode='both', names=names, epoch=0, it=0)

    #save ----------------
    print(all_accs.mean())
    with open(out_dir+'/valid/results-summary.txt', 'w') as f:
        for n in range(num_valid):
            f.write('%d\t%s\t%f\n'%(all_indices[n],names[n],all_accs[n]))




    pass



def run_make_priors():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-weight-00'
    #out_dir  = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-00'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/valid/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/make_validate.code.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    def valid_augment(image,label):
        #image, label = random_horizontal_flip2(image, label,u=1)
        # image, label = random_shift_scale_rotate2(image, label, shift_limit=(-0.0625,0.0625),
        #           scale_limit=(-0.09,0.121), rotate_limit=(-0,0))

        #image, label = random_hue2(image, label, hue_limit=(-1,1), u=0.5)
        #image = random_hue(image, hue_limit=(-0.5,-0.5), u=1)
        #image = random_brightness(image, limit=(0.5,0.5), u=1)
        #image = random_contrast  (image, limit=(-0.5,-0.5), u=1)
        # image = random_saturation(image, limit=0.3, u=0.5)
        #image = random_gray(image, u=1)

        return  image, label





    batch_size   =  16
    valid_dataset = KgCarTrainDataset('train_5088', 'train128x128',  #'valid_v0_768',
                                transform=[
                                    lambda x,y:  valid_augment(x, y),
                                ])
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 6,
                        pin_memory  = True)
    ##check_dataset(valid_dataset, valid_loader), exit(0)

    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    num_valid = len(valid_dataset)
    names = valid_dataset.names
    all_accs   =np.zeros(num_valid,np.float32)
    all_indices=np.zeros(num_valid,np.float32)
    net.train()

    start=0
    end  =0
    for it, (images, labels, indices) in enumerate(valid_loader, 0):
        images  = Variable(images).cuda()
        labels  = Variable(labels).cuda()
        batch_size = len(indices)

        #forward
        logits = net(images)
        probs  = F.sigmoid(logits)
        masks  = (probs>0.5).float()

        loss = criterion(logits, labels)
        accs = dice_loss(masks, labels, is_average=False)
        acc  = accs.sum()/batch_size


        end = start + batch_size
        all_accs   [start:end]=accs.data.cpu().numpy()
        all_indices[start:end]=indices.cpu().numpy()
        start=end

        ## save ------------------------------------------------
        images = (images.data.cpu().numpy()*255).astype(np.uint8)
        images = np.transpose(images, (0, 2, 3, 1))
        probs  = (probs.data.cpu().numpy()*255).astype(np.uint8)
        batch_size,H,W,C = images.shape
        for b in range(batch_size):
            name = names[indices[b]]
            prob  = probs[b]
            im_show('prob',  prob,  resize=1)

            prior_dir = '/media/ssd/data/kaggle-carvana-cars-2017/priors/train128x128'
            cv2.imwrite(prior_dir+'/%s.png'%(name), prob)

            cv2.waitKey(1)


    #save ----------------
    print(all_accs.mean())




    pass

# ------------------------------------------------------------------------------------

def run_make_probs():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-weight-00'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/make_probs.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 8

    test_dataset = KgCarTestDataset( 'test_100064',  #100064  #3197
                                     'test128x128',
                                     transform= [
                                    ],)
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
    if 1:
        #start = timer()
        #log.write('\tnp.load time = %f min\n'%((timer() - start) / 60))

        start = timer()
        net.eval()
        probs = predict8_in_blocks( net, test_loader, block_size=CARVANA_BLOCK_SIZE, save_dir=out_dir+'/submit',log=log)           # 20 min
        log.write('\tpredict_in_blocks = %f min\n'%((timer() - start) / 60))

    #probs = probs.astype(np.float32)/255
    log.write('\n')
    exit(0)


def run_submit():

    out_dir = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-weight-00'

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')


    # read names
    split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    names = [name+'.jpg' for name in names]
    num_test = len(names)


    rles=[]
    for i in range(CARVANA_NUM_BLOCKS):
        start = timer()
        ps   = np.load(out_dir+'/submit/probs-part%02d.8.npy'%i)
        inds = np.loadtxt(out_dir+'/submit/indices-part%02d.8.txt'%i,dtype=np.int32)
        log.write('\tnp.load time = %f min\n'%((timer() - start) / 60))

        M = len(ps)
        for m in range(M):
            if (m%1000==0):
                n = len(rles)
                end  = timer()
                time = (end - start) / 60
                time_remain = (num_test-n-1)*time/(n+1)
                print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
            #--------------------------------------------------------
            p=ps[m]
            ind=inds[m]

            #project back to
            #prob = np.zeros((CARVANA_HEIGHT,CARVANA_WIDTH), np.uint8)
            prob = cv2.resize(p,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT))
            mask = prob>127
            rle = run_length_encode(mask)
            rles.append(rle)


            #debug
            if 0:
                name = names[ind]
                img_file = CARVANA_DIR + '/images/test/%s'%(name)
                image = cv2.imread(img_file)
                results = make_combined_image(image, label=None, prob=prob)
                im_show('results',results,0.33)
                im_show('prob',prob,0.33)
                im_show('p',p,0.33)
                cv2.waitKey(0)

    #-----------------------------------------------------
    start = timer()
    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit/results-%s.csv.gz'%dir_name
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')
    log.write('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #3 min
    log.write('\n')






# ------------------------------------------------------------------------------------
# https://www.kaggle.com/tunguz/baseline-2-optimal-mask/code
def run_check1():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024_crop00_01d'
    model_file = out_dir +'/snap/final.pth'  #final
    os.makedirs(out_dir+'/valid/results',  exist_ok=True)

    test_dataset = KgCarTestDataset( 'valid_v0_768',
                                transform=[
                                    lambda x:  fix_crop(x, roi=(0,0,1024,1024)),
                                ])

    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 8,
                        drop_last   = False,
                        num_workers = 5,
                        pin_memory  = True)

    H,W = CARVANA_IMAGE_H, CARVANA_IMAGE_W

    ## net ----------------------------------------
    net = Net(in_shape=(3, H, W), num_classes=1)
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    print('start prediction ...')
    if 1:
        net.eval()
        probs = predict8( net, test_loader )
        show_test_results(test_dataset, probs, N=100, wait=1, save_dir=out_dir+'/valid/results')

    pass



#decode and check
def run_check_submit_csv():

    #gz_file ='/root/share/project/kaggle-carvana-cars/results/ensemble/xxx/submit/results-xxx.csv.gz'
    gz_file ='/root/share/project/kaggle-carvana-cars/results/__old_2__/UNet_double_1024_5/submit/results1-0.996.csv.gz'
    df = pd.read_csv(gz_file, compression='gzip', error_bad_lines=False)
    #save_dir ='/root/share/project/kaggle-carvana-cars/results/ensemble/xxx/submit/results'
    #os.makedirs(save_dir,  exist_ok=True)

    if 0:
        def fix_string(string):
            string = string.replace('<replace>/','')
            string = string+'.jpg'
            return string
        df['img'] = df['img'].apply(fix_string)
        df.to_csv(gz_file, index=False, compression='gzip')

    indices = range(50000)  #[0,1,2,32000-1,32000,32000+1,100064-1]
    for n in indices:
        name = df.values[n][0]
        img_file = CARVANA_DIR + '/images/test/%s'%(name)
        image = cv2.imread(img_file)

        rle   = df.values[n][1]
        mask  = run_length_decode(rle,H=CARVANA_HEIGHT, W=CARVANA_WIDTH)
        results = make_combined_image(image, label=None, prob=mask)
        draw_shadow_text  (results, '%06d: %s'%(n,name), (5,80),  2, (255,255,255), 4)

        im_show('mask', mask, resize=0.25)
        im_show('results',results,0.10)
        cv2.waitKey(0)
        #cv2.imwrite(save_dir + '/%s'%name, results)

    pass

def run_make_priors1():

    out_dir    = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-00d'

    #logging, etc --------------------

    # read names
    split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    num_test = len(names)

    CSV_BLOCK_SIZE= 16000
    num_blocks = int(math.ceil(num_test/CSV_BLOCK_SIZE))
    for i in range(num_blocks):
        start = timer()
        ps   = np.load(out_dir+'/submit/probs-part%02d.8.npy'%i)
        inds = np.loadtxt(out_dir+'/submit/indices-part%02d.8.txt'%i,dtype=np.int32)
        #log.write('\tnp.load time = %f min\n'%((timer() - start) / 60))

        M = len(ps)
        for m in range(M):
            # if (m%1000==0):
            #     n = len(rles)
            #     end  = timer()
            #     time = (end - start) / 60
            #     time_remain = (num_test-n-1)*time/(n+1)
            #     print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
            # #--------------------------------------------------------
            p=ps[m]
            ind=inds[m]

            #project back to
            prob = p #cv2.resize(p,dsize=(1024,1024),interpolation=cv2.INTER_LINEAR)


            view = int(names[ind][-2:])
            if view==2:
                prob = do_post_process1(p)


            name = names[ind]
            prior_dir ='/media/ssd/data/kaggle-carvana-cars-2017/priors/test1024x1024_1'
            cv2.imwrite(prior_dir + '/%s.png'%name,prob)

            if 0:
                img_file = CARVANA_DIR + '/images/test/%s.jpg'%(name)
                #img_file = CARVANA_DIR + '/images/train/%s'%(name)
                image = cv2.imread(img_file)
                image=cv2.resize(image,dsize=(1024,1024))
                results = make_results_image(image, label=None, prob=prob)
                im_show('results',results,0.33)
                im_show('prob',prob,0.33)
                cv2.waitKey(0)
    #-----------------------------------------------------
    start = timer()


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_make_probs()
    #run_submit()

    #run_train()


    #run_check_submit_csv()
    #run_make_validate_results()

    run_make_priors1()
    print('\nsucess!')