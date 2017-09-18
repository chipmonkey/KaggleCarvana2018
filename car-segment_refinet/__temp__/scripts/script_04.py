# find the dice loss limit of resizing
#import sys
#sys.path.append('/share/project/kaggle-carvana-cars/build/car-segment_single')
#print(sys.path)

from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py


# try and understand different kinds of convolutions

def try_convolution():
    H,W = 8,8
    input = np.zeros((1,2,H,W), np.float32)
    input[0,0,4,4]=1
    input[0,0,4,5]=1

    w = np.array([[[[1,2,3],[0,0,0],[0,0,0]]]], np.float32)


    if 0:
        for x in range(W):
            for y in range(H):
                if x%4==0 and y%4==0:
                    input[0,y,x]=1

    conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1, bias=False)
    #conv.weight.data = torch.from_numpy(w)
    ## initialistaion
    #  https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    #  bang liu : random normal(mean=0,std=filter_width*filter_height*channel).

    # for m in self.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))

    input = Variable(torch.from_numpy(input))
    output = conv(input)
    output = output.data
    print('input',input)
    print('output',output)
    print('conv.weight',conv.weight)


    pass



def try_deconvolution():
    H,W = 4,4
    input = np.zeros((1,2,H,W), np.float32)
    input[0,0,2,2]=1
    #input[0,0,3,2]=10

    w = np.array([[[[1,2,3],[0,0,0],[0,0,0]]]], np.float32)


    if 0:
        for x in range(W):
            for y in range(H):
                if x%4==0 and y%4==0:
                    input[0,y,x]=1

    conv = nn.ConvTranspose2d(2, 1, kernel_size=4, padding=1, stride=2, bias=False)
    #conv.weight.data = torch.from_numpy(w)
    ## initialistaion
    #  https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    #  bang liu : random normal(mean=0,std=filter_width*filter_height*channel).

    # for m in self.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))

    input = Variable(torch.from_numpy(input))
    output = conv(input)
    output = output.data
    print('input',input)
    print('output',output)
    print('conv.weight',conv.weight)


    pass
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    try_deconvolution()

    print('\nsucess!')