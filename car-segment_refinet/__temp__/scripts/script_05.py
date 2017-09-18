
from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py


# try spatial normlisation
#    https://arxiv.org/pdf/1301.2820.pdf
#    https://github.com/pytorch/pytorch/blob/master/torch/legacy/nn/SpatialSubtractiveNormalization.py
#    https://github.com/pytorch/pytorch/issues/653
#    https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/thnn/normalization.py

#    https://github.com/pytorch/pytorch/tree/master/torch/lib/THCUNN

from torch.legacy.nn import SpatialCrossMapLRN
from torch.legacy.nn import SpatialSubtractiveNormalization

def try_lrn_0():
    img_file='/media/ssd/data/kaggle-carvana-cars-2017/images/train640x896/0cdf5b5d0ce1_03.jpg'
    img=cv2.imread(img_file)

    image = img.astype(np.float32)/255
    image = torch.unsqueeze(image_to_tensor(image),0)
    x = Variable(image)

    #lrn = SpatialCrossMapLRN(size=5, alpha=1e-4, beta=0.75, k=1)
    lrn = SpatialSubtractiveNormalization( nInputPlane=3, kernel=torch.Tensor(11, 11).fill_(1))##9, 9
    y = lrn.updateOutput(x.data)
    #y = y.data
    y = torch.clamp(0.5*(y+1),0,1)



    x = torch.squeeze(x.data)
    y = torch.squeeze(y)
    img = tensor_to_image(x,std=255)
    res = tensor_to_image(y,std=255)
    im_show('img', img, resize=0.5)
    im_show('res', res, resize=0.5)
    cv2.waitKey(0)

    pass

def try_lrn_1():
    img_file='/media/ssd/data/kaggle-carvana-cars-2017/images/train640x896/0cdf5b5d0ce1_03.jpg'
    img=cv2.imread(img_file)

    image = img.astype(np.float32)/255
    image = torch.unsqueeze(image_to_tensor(image),0)
    x = Variable(image).cuda()

    #lrn = SpatialCrossMapLRN(size=5, alpha=1e-4, beta=0.75, k=1)
    lrn = SpatialSubtractiveNormalization( nInputPlane=3, kernel=torch.Tensor(11, 11).fill_(1))##9, 9
    y = lrn.updateOutput(x.data)
    #y = y.data
    y = torch.clamp(0.5*(y+1),0,1)


    x = x.cpu()
    x = torch.squeeze(x.data.cpu())
    y = torch.squeeze(y)
    img = tensor_to_image(x,std=255)
    res = tensor_to_image(y,std=255)
    im_show('img', img, resize=0.5)
    im_show('res', res, resize=0.5)
    cv2.waitKey(0)

    pass



def try_lrn():
    img_file='/media/ssd/data/kaggle-carvana-cars-2017/images/train640x896/0cdf5b5d0ce1_03.jpg'
    img=cv2.imread(img_file)

    image = img.astype(np.float32)/255
    image = torch.unsqueeze(image_to_tensor(image),0)
    x = Variable(image).cuda()

    if 1: #simply zero mean and divide by std
        z = torch.unsqueeze(torch.sum(x,dim=1),1)
        z2_mean = F.avg_pool2d(z*z,kernel_size=7,padding=3,stride=1)
        z_mean  = F.avg_pool2d(z,  kernel_size=7,padding=3,stride=1)
        z_std   = torch.sqrt(z2_mean-z_mean*z_mean+0.03)
        z = (z-z_mean)/(z_std)
        y = torch.clamp(0.5*(z+1),0,1)
    if 0: #simply zero mean and divide by std
        x2_mean = F.avg_pool2d(x*x,kernel_size=11,padding=5,stride=1)
        x_mean  = F.avg_pool2d(x,  kernel_size=11,padding=5,stride=1)
        x_std   = torch.sqrt(x2_mean-x_mean*x_mean+0.01)

        y = (x-x_mean)/(x_std)
        y = torch.unsqueeze(torch.sum(y,dim=1),0)/3
        y = torch.clamp(0.5*(y+1),0,1)

    if 0: #lrn
        alpha=1e-4 * 11*11
        beta=0.75
        k=1

        x2 = x*x
        x2_mean = F.avg_pool2d(x2,kernel_size=11,padding=5,stride=1)
        y = x/torch.pow(k+alpha*x2_mean,beta)
        y = torch.clamp(0.5*(y+1),0,1)

    #----------------------------------------------
    x = x.cpu().data
    y = y.cpu().data
    x = x[0]
    y = y[0]
    img = tensor_to_image(x,std=255)
    res = tensor_to_image(y,std=255)
    im_show('img', img, resize=0.5)
    im_show('res', res, resize=0.5)
    cv2.waitKey(0)

    pass


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    try_lrn()

    print('\nsucess!')