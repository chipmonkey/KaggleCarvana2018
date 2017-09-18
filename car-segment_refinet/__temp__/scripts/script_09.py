
from common import *
from dataset.carvana_cars import *
from net.tool import *

from net.segmentation.loss import *
from net.segmentation.my_unet_baseline import UNet1024 as Net
from net.segmentation.blocks import *


# timing test ###
def run_time():

    is_fp16     = 1
    is_merge_bn = 0

    out_dir    = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-00b'
    model_file = out_dir +'/snap/064.pth'  #final

    ## dataset ----------------------------
    batch_size = 8

    test_dataset = KgCarDataset( #'test_100064',  'test1024x1024',#100064  #3197
                                 'valid_v0_768',  'train1024x1024',#100064  #3197
                                 transform=[], mode='train')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 12,
                        pin_memory  = True)
    test_iter = iter(test_loader)

    ## net ----------------------------------------
    net = Net(in_shape = (3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda()


    if is_merge_bn: merge_bn_in_net(net)
    if is_fp16:     net.half()


    ## start testing now #####
    net.eval()

    start_time = timer()
    num        = 0
    accuracy   = 0
    while True:

        try:
            images, labels, indices = test_iter.next()
        except StopIteration:
            break

        # forward
        if is_fp16:
            images = Variable(images,volatile=True).cuda().half()
        else:
            images = Variable(images,volatile=True).cuda()

        labels = Variable(labels).cuda()
        logits = net(images)
        probs  = F.sigmoid(logits)


        batch_size = len(indices)
        probs  = (probs.float()>0.5).float()
        l = dice_loss(probs,labels)
        l = l.data.cpu().numpy()[0]
        accuracy += l*batch_size
        num  += batch_size

        print('num=%05d,  accuracy=%f'%(num,accuracy))
    pass
    end_time   = timer()
    time_taken = end_time - start_time
    time_taken = time_taken

    accuracy   = accuracy/num
    print('')
    print('time_taken=%0.2f  sec'%time_taken)
    print('accuracy=%0.6f'%accuracy)
    print('is_fp16=%s'%str(is_fp16))
    print('is_merge_bn=%s'%str(is_merge_bn))
    print('batch_size=%d'%batch_size)


'''
time_taken=45.33  sec
accuracy=0.996709
is_fp16=0
is_merge_bn=0
batch_size=8

time_taken=45.08  sec
accuracy=0.996709
is_fp16=0
is_merge_bn=1
batch_size=8

time_taken=43.74  sec
accuracy=0.996710
is_fp16=1
is_merge_bn=1
batch_size=8

time_taken=43.42  sec
accuracy=0.996710
is_fp16=1
is_merge_bn=1
batch_size=16

time_taken=47.98  sec
accuracy=0.996709
is_fp16=1
is_merge_bn=0
batch_size=16


time_taken=51.90  sec
accuracy=0.996710
is_fp16=1
is_merge_bn=1
batch_size=8




'''


# main #################################################################

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_time()

    print('sucess!')
