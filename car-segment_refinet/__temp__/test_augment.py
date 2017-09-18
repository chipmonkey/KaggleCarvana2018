from common import *
from submit import *
from dataset.carvana_cars import *

from net.tool import *
from net.rate import *
from net.segmentation.loss import *
#from net.segmentation.my_unet import UNet1024 as Net
from net.segmentation.my_unet_baseline import UNet1024 as Net
#from net.segmentation.my_unet import MoreUNet as Net
#from net.segmentation.my_frrunet import FRRUNet512 as Net


#from net.segmentation.my_unet_baseline import UNet128 as Net
#from net.segmentation.my_unet_baseline import UNet1024 as Net
#from net.segmentation.my_unet_baseline import UNet512 as Net
#from net.segmentation.my_unet import LargeUNet128 as Net
#from net.segmentation.my_unet import LessUNet512 as Net

## ----------------------------------------------------------------------------------
#  ffmpeg -y -loglevel 0 -f image2 -r 15 -i 'rcnn_post_nms/%*.jpg' -b:v 8000k rcnn_post_nms.avi
#  ffmpeg -i results.avi -vf scale=300:100 -b:v 8000k results-small.avi



## experiment setting here ----------------------------------------------------
def criterion(logits, labels, is_weight=True):
    #l = BCELoss2d()(logits, labels)
    #l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)

    #compute weights
    btach_size,H,W = labels.size()
    if   H == 128: kernel_size =11
    elif H == 256: kernel_size =21
    elif H == 512: kernel_size =21
    elif H == 1024: kernel_size=41
    else: raise ValueError('exit at criterion()')

    a   = F.avg_pool2d(labels,kernel_size=kernel_size,padding=kernel_size//2,stride=1)
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
def dice_loss(m1, m2, is_average=True):
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
            #os.makedirs(out_dir+'/results_by_name', exist_ok=True)
            #os.makedirs(out_dir+'/results_by_score', exist_ok=True)
            #cv2.imwrite(save_dir + '/%05d-%03d.jpg'%(it,b), results)
            if mode in['both','by_score'] : cv2.imwrite(out_dir+'/results_by_score/%0.5f-%s.png'%(score,name), results)
            if mode in['both','by_name' ] : cv2.imwrite(out_dir+'/results_by_name/%s.png'%(name), results)

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


#  100064/32000 =
def predict8_in_blocks(net, test_loader, block_size=CARVANA_BLOCK_SIZE, log=None, save_dir=None, backward_transform=(lambda x:x)):

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
            p  [m : m+batch_size] = backward_transform(probs)
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
    for it, (images, labels, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)
        labels = Variable(labels.cuda(),volatile=True)

        # forward
        logits = net(images)
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



# ------------------------------------------------------------------------------------
def run_train():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-shallow-01a'
    #out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet128-weight-id2-labels-prior-02a'
    initial_checkpoint = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-shallow-01/034.pth'
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
    def train_augment(image,label):
        image, label = random_horizontal_flipN([image, label])
        image, label = random_shift_scale_rotateN([image, label], shift_limit=(-0.0625,0.0625),
                  scale_limit=(-0.09,0.121), rotate_limit=(-0,0))

        #image, label = random_mask_hue(image, label, hue_limit=(-1,1), u=0.5)
        #image = random_hue(image, hue_limit=(-1,1), u=0.5)
        image = random_brightness(image, limit=(-0.5,0.5), u=0.5)
        image = random_contrast  (image, limit=(-0.5,0.5), u=0.5)
        image = random_saturation(image, limit=(-0.3,0.3), u=0.5)
        image = random_gray(image, u=0.25)

        return  image, label

    log.write('** dataset setting **\n')
    batch_size   =  3
    num_grad_acc =  15//batch_size


    train_dataset = KgCarDataset(  'train_v0_4320', 'train1024x1024', ## 'train_160', ##128x128 ##256x256  #train_v0_4320
                                transform = [ lambda x,y:train_augment(x,y), ], mode='train')
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True)
    ##check_dataset(train_dataset, train_loader), exit(0)

    valid_dataset = KgCarDataset('valid_v0_768', 'train1024x1024',
                                transform=[], mode='train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 6,
                        pin_memory  = True)


    test_dataset = KgCarDataset( 'test_3197', 'test',
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

    net = Net(in_shape=(3, 128, 128))
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
    #LR = StepLR([ (0, 0.01),  (35, 0.005),  (40,0.001),  (42, -1),(44, -1)]) #bn mode
    #LR = StepLR([ (0, 0.01),  (10, 0.005),  (40,0.001),  (42, 0.0001),(44, -1)]) #bn mode
    LR = StepLR([ (0, 0.01),  (40, 0.005),  (45,0.001),  (47, -1),(44, -1)]) #bn mode

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
        for it, (images, labels, indices) in enumerate(train_loader, 0):
            images  = Variable(images).cuda()
            labels  = Variable(labels).cuda()

            #forward
            logits = net(images)
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

def run_make_results():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-weight-00'
     #out_dir  = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-00'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    shutil.rmtree(out_dir+'/valid/results_by_score',ignore_errors=True)
    shutil.rmtree(out_dir+'/valid/results_by_name',ignore_errors=True)
    os.makedirs(out_dir+'/valid/results_by_score',  exist_ok=True)
    os.makedirs(out_dir+'/valid/results_by_name',   exist_ok=True)
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





    batch_size   =  1
    valid_dataset = KgCarDataset(
                                #'train_v0_4320', 'train128x128',
                                'valid_v0_768', 'train512x512',
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
    net.eval()

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

        # show_batch_results(indices, images, probs, labels=labels, wait=1,
        #                    out_dir=out_dir+'/valid', mode='both', names=names, epoch=0, it=0)

    #save ----------------
    accuracy = all_accs.mean()
    print('accuracy=%f'%accuracy)
    with open(out_dir+'/valid/results-summary.txt', 'w') as f:
        for n in range(num_valid):
            f.write('%d\t%s\t%f\n'%(all_indices[n],names[n],all_accs[n]))

        f.write('\naccuracy=%f\n'%accuracy)


    pass



def run_make_full_results():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00'
     #out_dir  = '/root/share/project/kaggle-carvana-cars/results/baseline/UNet128-00'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    shutil.rmtree(out_dir+'/valid/full_results_by_score',ignore_errors=True)
    ##shutil.rmtree(out_dir+'/valid/full_results_by_name',ignore_errors=True)
    os.makedirs(out_dir+'/valid/full_results_by_score',  exist_ok=True)
    os.makedirs(out_dir+'/valid/full_results_by_name',   exist_ok=True)
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





    batch_size   =  1
    valid_dataset = KgCarDataset(
                                #'train_v0_4320', 'train128x128',
                                'valid_v0_768', 'train1024x1024',
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
    net.eval()

    start=0
    end  =0
    for it, (images, labels, indices) in enumerate(valid_loader, 0):
        images  = Variable(images).cuda()
        labels  = Variable(labels).cuda()
        batch_size = len(indices)

        #forward
        logits = net(images)
        probs  = F.sigmoid(logits)


        probs  = (probs.data.cpu().numpy()*255).astype(np.uint8)
        for b in range(batch_size):
            name = names[indices[b]]
            mask_file = CARVANA_DIR + '/annotations/%s/%s_mask.png'%('train',name)
            label = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            label = label>128

            prob = probs[b]
            prob = cv2.resize(prob,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)  #INTER_CUBIC  ##
            mask = prob>128

            score = one_dice_loss_py(mask, label)
            all_accs   [start+b] = score
            all_indices[start+b] = indices[b]

            #----------------------------------------------------------------
            if 1:
                results = np.zeros((CARVANA_HEIGHT*CARVANA_WIDTH,3),np.uint8)
                a = (2*label+mask).reshape(-1)
                miss = np.where(a==2)[0]
                hit  = np.where(a==3)[0]
                fp   = np.where(a==1)[0]
                label_sum=label.sum()
                mask_sum =mask.sum()

                results[miss] = np.array([0,0,255])
                results[hit] = np.array([64,64,64])
                results[fp] = np.array([0,255,0])
                results = results.reshape(CARVANA_HEIGHT,CARVANA_WIDTH,3)
                L=30
                draw_shadow_text  (results, '%s.jpg'%(name), (5,1*L),  1, (255,255,255), 2)
                draw_shadow_text  (results, '%0.5f'%(score), (5,2*L),  1, (255,255,255), 2)
                draw_shadow_text  (results, 'label_sum  = %0.0f'%(label_sum), (5,3*L),  1, (255,255,255), 2)
                draw_shadow_text  (results, 'mask_sum  = %0.0f (%0.4f)'%(mask_sum,mask_sum/label_sum), (5,4*L),  1, (255,255,255), 2)
                draw_shadow_text  (results, 'miss  = %0.0f (%0.4f)'%(len(miss), len(miss)/label_sum), (5,5*L),  1, (0,0,255), 2)
                draw_shadow_text  (results, 'fp     = %0.0f (%0.4f)'%(len(fp), len(fp)/mask_sum), (5,6*L),  1, (0,255,0), 2)


                print(score)
                im_show('results',results,0.33)
                #im_show('label',label*255,0.33)
                #im_show('mask',mask*255,0.33)
                cv2.waitKey(1)

                cv2.imwrite(out_dir+'/valid/full_results_by_score/%0.5f-%s.png'%(score,name), results)
                cv2.imwrite(out_dir+'/valid/full_results_by_name/%s.png'%(name), results)
        start=start + batch_size


    #save ----------------
    accuracy = all_accs.mean()
    print('accuracy=%f'%accuracy)
    with open(out_dir+'/valid/full_results-summary.INTER_LINEAR.txt', 'w') as f:
        for n in range(num_valid):
            f.write('%d\t%s\t%f\n'%(all_indices[n],names[n],all_accs[n]))

        f.write('\naccuracy=%f\n'%accuracy)





    pass







# ------------------------------------------------------------------------------------

def run_make_probs():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-weight-00'
    model_file = out_dir +'/snap/042.pth'  #final

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

    test_dataset = KgCarDataset( 'valid_v0_768', 'train1024x1024',  #'test_100064', 'test512x512', #100064  #3197

                                     transform= [
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
    if 1:
        start = timer()
        net.eval()
        probs = predict8_in_blocks( net, test_loader, block_size=CARVANA_BLOCK_SIZE, save_dir=out_dir+'/submit',log=log)           # 20 min
        log.write('\tpredict_in_blocks = %f min\n'%((timer() - start) / 60))

    #probs = probs.astype(np.float32)/255
    log.write('\n')
    exit(0)

##############################################################################################################33

## possible augment: shift scale reflect
if 1:
    argument='none'
    def forward_transform(image):
        return image

    def backward_transform(probs):
        return probs


if 0:
    argument='reflect'
    def forward_transform(image):
        image = cv2.flip(image,1)
        return image

    def backward_transform(probs):
        N,H,W = probs.shape
        probs = probs.reshape(N*H,W)
        probs = cv2.flip(probs,1)
        probs = probs.reshape(N,H,W)
        return probs

if 0:
    scale = 1
    dx,dy = 0,0

    # argument='scale1.10'
    # scale = 1.10
    #
    # argument='scale0.90909'
    # scale = 0.90909

    # argument='right25'
    # dx,dy = 25,25

    argument='left25'
    dx,dy = -25,-25


    def forward_transform(image):
        #image = cv2.resize(image,dsize=(1024,1024),fx=1.10,fy=1.10)
        #im_show('image_before',image*255,0.33)
        #cv2.waitKey(1)

        width,height=1024,1024
        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,np.eye(2,dtype=np.float32)*scale) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat   = cv2.getPerspectiveTransform(box0,box1)
        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        #im_show('image_after',image*255,0.33)
        #cv2.waitKey(0)
        return image

    def backward_transform(probs):
        N,H,W = probs.shape

        width,height=W,H
        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,np.eye(2,dtype=np.float32)/scale) + np.array([width/2-dx,height/2-dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat   = cv2.getPerspectiveTransform(box0,box1)

        for n in range(N):
            probs[n] = cv2.warpPerspective(probs[n], mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        return probs




def run_make_augment_probs():
    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-shallow-01b'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit_%s/results'%argument,  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/make_probs.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit_%s.txt'%argument,mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 1

    test_dataset = KgCarDataset( #'test_100064', 'test512x512', #100064  #3197
                                 'valid_v0_768',  'train1024x1024',
                                     transform= [
                                         lambda x:  forward_transform(x),
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
    if 1:
        start = timer()
        net.eval()
        probs = predict8_in_blocks( net, test_loader, block_size=CARVANA_BLOCK_SIZE,
                                    save_dir=out_dir+'/submit_%s'%argument, log=log,
                                    backward_transform=backward_transform)           # 20 min
        log.write('\tpredict_in_blocks = %f min\n'%((timer() - start) / 60))

    #probs = probs.astype(np.float32)/255
    log.write('\n')




def run_augment_submit():
    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-shallow-01b'

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit_%s/results'%argument,  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit_%s.txt'%argument,mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')


    # read names
    split_file = CARVANA_DIR +'/split/'+ 'valid_v0_768'
    CARVANA_NUM_BLOCKS =1


    #split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    names = [name+'.jpg' for name in names]
    num_test = len(names)


    rles=[]
    for i in range(CARVANA_NUM_BLOCKS):
        start = timer()
        ps   = np.load(out_dir+'/submit_%s/probs-part%02d.8.npy'%(argument,i))
        inds = np.loadtxt(out_dir+'/submit_%s/indices-part%02d.8.txt'%(argument,i),dtype=np.int32)
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
            prob = cv2.resize(p,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)
            mask = prob>127
            rle = run_length_encode(mask)
            rles.append(rle)


            #debug
            if m<10 and i==0:
                name = names[ind]
                #img_file = CARVANA_DIR + '/images/test/%s'%(name)
                img_file = CARVANA_DIR + '/images/train/%s'%(name)
                image = cv2.imread(img_file)
                results = make_combined_image(image, label=None, prob=prob)
                im_show('results',results,0.33)
                im_show('prob',prob,0.33)
                im_show('p',p,0.33)
                cv2.waitKey(1)

    #-----------------------------------------------------
    start = timer()
    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit_%s/results-%s.csv.gz'%(argument,dir_name)
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')
    log.write('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #3 min
    log.write('\n')


def run_accuracy_from_csv():

    #gz_file ='/root/share/project/kaggle-carvana-cars/results/ensemble/xxx/submit/results-xxx.csv.gz'
    #gz_file  ='/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/submit/results-UNet1024-weight-00.csv.gz'
    ##gz_file  ='/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/submit_reflect/results-UNet1024-weight-00.csv.gz'
    #gz_file  ='/root/share/project/kaggle-carvana-cars/results/single/UNet1024-shallow-01b/submit_%s/results-UNet1024-shallow-01b.csv.gz'%argument


    gz_file  ='/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/more-results-UNet1024-weight-00.csv.gz'
    #------------------------------------------------------------------------
    df = pd.read_csv(gz_file, compression='gzip', error_bad_lines=False)


    #read images
    split_file = CARVANA_DIR +'/split/'+ 'valid_v0_768'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    #names = [name+'.jpg' for name in names]
    num_test = len(names)


    all_score = 0
    for n in range(num_test):
        if(n%1000==0): print(n)
        name = names[n]
        assert(name+'.jpg' == df.values[n][0])
        rle  = df.values[n][1]


        mask = run_length_decode(rle,H=CARVANA_HEIGHT, W=CARVANA_WIDTH,fill_value=255)
        mask = (mask>127).astype(np.uint8)

        mask_file = CARVANA_DIR + '/annotations/%s/%s_mask.png'%('train',name)
        label = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
        label = (label>127).astype(np.uint8)


        score = one_dice_loss_py(label,mask)
        print('%0.6f'%score)
        all_score += score

        if 0:
            img_file = CARVANA_DIR + '/images/train/%s.jpg'%(name)
            image = cv2.imread(img_file)
            label=label*255
            mask =mask*255
            results = make_combined_image(image, label=label, prob=mask)
            im_show('results',results,0.33)
            im_show('mask',  mask,0.33)
            im_show('label',label,0.33)
            cv2.waitKey(0)

    pass
    print('final')
    print('%f\n %0.6f'%(all_score, all_score/num_test))





def run_ensemble_submit():

    gz_file  ='/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/more-results-UNet1024-weight-00.csv.gz'


    arugment_files =[
        '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/submit/results-UNet1024-weight-00.csv.gz',
        # '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/submit_reflect/results-UNet1024-weight-00.csv.gz',
        # '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/submit_scale1.10/results-UNet1024-weight-00.csv.gz',
        # '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/submit_scale0.90909/results-UNet1024-weight-00.csv.gz',
        # '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/submit_right25/results-UNet1024-weight-00.csv.gz',
        # '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-weight-00/enesmble/submit_left25/results-UNet1024-weight-00.csv.gz',
        '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-shallow-01b/submit_%s/results-UNet1024-shallow-01b.csv.gz'%argument

    ]
    num_arugments = len(arugment_files)

    dfs=[]
    for file in arugment_files:
        df = pd.read_csv(file, compression='gzip', error_bad_lines=False)
        dfs.append(df)

    #----------------------------------------------------------------------------------------------------------------------------------


    # read names
    split_file = CARVANA_DIR +'/split/'+ 'valid_v0_768'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    names = [name+'.jpg' for name in names]
    num_test = len(names)

    rles = []
    for n in range(num_test):
        if(n%1000==0): print(n)

        name = dfs[0].values[n][0]
        mask = np.zeros((CARVANA_HEIGHT,CARVANA_WIDTH),np.uint8)

        for df in dfs:
            assert(name == df.values[n][0])
            r = df.values[n][1]
            m = run_length_decode(r,H=CARVANA_HEIGHT, W=CARVANA_WIDTH,fill_value=1)
            mask = mask + m

        mask = mask>num_arugments*0.1
        rle = run_length_encode(mask)
        rles.append(rle)

        #debug
        if 0:
            #img_file = CARVANA_DIR + '/images/test/%s'%(name)
            img_file = CARVANA_DIR + '/images/train/%s'%(name)
            image = cv2.imread(img_file)
            prob  = (mask*255).astype(np.uint8)
            results = make_combined_image(image, label=None, prob=prob)
            im_show('results',results,0.33)
            im_show('prob',prob,0.33)
            cv2.waitKey(0)
    pass

    start = timer()
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')
    print('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #3 min




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_ensemble_submit()
    #run_make_augment_probs()
    #run_augment_submit()
    run_accuracy_from_csv()
    # print(argument)

    #

    print('\nsucess!')