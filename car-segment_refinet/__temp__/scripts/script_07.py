
from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py
from net.segmentation.loss import *

# check downsample size
'''
x     torch.Size([1, 3, 640, 640])
down1 torch.Size([1, 32, 640, 640])
down2 torch.Size([1, 64, 320, 320])
down3 torch.Size([1, 128, 160, 160])
down4 torch.Size([1, 256, 80, 80])
down5 torch.Size([1, 512, 40, 40])
down6 torch.Size([1, 1024, 20, 20])
out   torch.Size([1, 1024, 10, 10])


x     torch.Size([1, 3, 640, 960])
down1 torch.Size([1, 32, 640, 960])
down2 torch.Size([1, 64, 320, 480])
down3 torch.Size([1, 128, 160, 240])
down4 torch.Size([1, 256, 80, 120])
down5 torch.Size([1, 512, 40, 60])
down6 torch.Size([1, 1024, 20, 30])
out   torch.Size([1, 1024, 10, 15])

'''


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    try_pytorch_upsample_error()

    print('\nsucess!')