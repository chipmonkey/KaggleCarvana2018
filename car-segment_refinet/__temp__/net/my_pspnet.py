# # top segmentation arhitecture: http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6
# # http://image-net.org/challenges/ilsvrc+coco2016
#
# # pspnet from scratch (Pyramid Scene Parsing Network)
# #  https://arxiv.org/abs/1612.01105
# #  https://medium.com/@steve101777/dense-segmentation-pyramid-scene-parsing-pspnet-753b1cb6097c
# #  https://github.com/hszhao/PSPNet
# #  https://raw.githubusercontent.com/hszhao/PSPNet/master/evaluation/prototxt/pspnet101_VOC2012_473.prototxt
#
#
# # http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
#
#
# from common import *
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# #  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
# #  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average)
#
#     def forward(self, logits, targets):
#         return self.nll_loss(F.log_softmax(logits), targets)
#
# class BCELoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(BCELoss2d, self).__init__()
#         self.bce_loss = nn.BCELoss(weight, size_average)
#
#     def forward(self, logits, targets):
#         probs        = F.sigmoid(logits)
#         probs_flat   = probs.view (-1)
#         targets_flat = targets.view(-1)
#         return self.bce_loss(probs_flat, targets_flat)
#
# ## -------------------------------------------------------------------------------------
#
# def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#     return [
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#     ]
#
#
# class UNet128 (nn.Module):
#
#     def __init__(self, in_shape, num_classes):
#         super(UNet128, self).__init__()
#         in_channels, height, width = in_shape
#
#         #128
#
#         self.down1 = nn.Sequential(
#             *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(16, 32, kernel_size=1, stride=1, padding=0 ),
#         )
#         #64
#
#         self.down2 = nn.Sequential(
#             *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0 ),
#         )
#         #32
#
#         self.down3 = nn.Sequential(
#             *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0 ),
#         )
#         #16
#
#         self.down4 = nn.Sequential(
#             *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
#         )
#         #8
#
#         self.same = nn.Sequential(
#             *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
#         )
#
#         #16
#         self.up4 = nn.Sequential(
#             *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
#             #nn.Dropout(p=0.10),
#         )
#         #16
#
#         self.up3 = nn.Sequential(
#             *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
#         )
#         #32
#
#         self.up2 = nn.Sequential(
#             *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
#         )
#         #64
#
#         self.up1 = nn.Sequential(
#             *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
#         )
#         #128
#
#         self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )
#
#
#
#     def forward(self, x):
#
#         #128
#
#         down1 = self.down1(x)
#         out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64
#
#         down2 = self.down2(out)
#         out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32
#
#         down3 = self.down3(out)
#         out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16
#
#         down4 = self.down4(out)
#         out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8
#
#         out   = self.same(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #16
#         out   = torch.cat([down4, out],1)
#         out   = self.up4(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #32
#         out   = torch.cat([down3, out],1)
#         out   = self.up3(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #64
#         out   = torch.cat([down2, out],1)
#         out   = self.up2(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #128
#         out   = torch.cat([down1, out],1)
#         out   = self.up1(out)
#         out   = self.classify(out)
#         #out   = F.sigmoid(out)
#
#         return out
#
#
#
# # a bigger version for 256x256
# class UNet256 (nn.Module):
#
#     def __init__(self, in_shape, num_classes):
#         super(UNet256, self).__init__()
#         in_channels, height, width = in_shape
#
#         #256
#
#         self.down1 = nn.Sequential(
#             *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(16, 32, kernel_size=1, stride=2, padding=0 ),
#         )
#         #64
#
#         self.down2 = nn.Sequential(
#             *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0 ),
#         )
#         #32
#
#         self.down3 = nn.Sequential(
#             *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0 ),
#         )
#         #16
#
#         self.down4 = nn.Sequential(
#             *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
#             *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
#         )
#         #8
#
#         self.same = nn.Sequential(
#             *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
#         )
#
#         #16
#         self.up4 = nn.Sequential(
#             *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
#             #nn.Dropout(p=0.10),
#         )
#         #16
#
#         self.up3 = nn.Sequential(
#             *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
#         )
#         #32
#
#         self.up2 = nn.Sequential(
#             *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
#         )
#         #64
#
#         self.up1 = nn.Sequential(
#             *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0 ),
#             *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
#         )
#         #128
#
#         self.up0 = nn.Sequential(
#             *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
#         )
#         #256
#
#         self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )
#
#
#
#     def forward(self, x):
#
#         #256
#
#         down1 = self.down1(x)
#         out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64
#
#         down2 = self.down2(out)
#         out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32
#
#         down3 = self.down3(out)
#         out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16
#
#         down4 = self.down4(out)
#         out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8
#
#         out   = self.same(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #16
#         out   = torch.cat([down4, out],1)
#         out   = self.up4(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #32
#         out   = torch.cat([down3, out],1)
#         out   = self.up3(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #64
#         out   = torch.cat([down2, out],1)
#         out   = self.up2(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #128
#         out   = torch.cat([down1, out],1)
#         out   = self.up1(out)
#
#         out   = F.upsample_bilinear(out, scale_factor=2) #128
#         out   = self.up0(out)
#
#         out   = self.classify(out)
#
#         return out
#
#
#
# # main #################################################################
# if __name__ == '__main__':
#     print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#     batch_size  = 32
#     C,H,W = 3,256,256
#
#     if 0: # CrossEntropyLoss2d()
#         inputs = torch.randn(batch_size,C,H,W)
#         labels = torch.LongTensor(batch_size,H,W).random_(1)
#
#         net = UNet(in_shape=(C,H,W), num_classes=2).cuda().train()
#         x = Variable(inputs).cuda()
#         y = Variable(labels).cuda()
#         logits = net.forward(x)
#
#         loss = CrossEntropyLoss2d()(logits, y)
#         loss.backward()
#
#         print(type(net))
#         print(net)
#
#         print('logits')
#         print(logits)
#
#
#
#     if 1: # BCELoss2d()
#         num_classes = 1
#
#         inputs = torch.randn(batch_size,C,H,W)
#         labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)
#
#         net = UNet256(in_shape=(C,H,W), num_classes=1).cuda().train()
#         x = Variable(inputs).cuda()
#         y = Variable(labels).cuda()
#         logits = net.forward(x)
#
#         loss = BCELoss2d()(logits, y)
#         loss.backward()
#
#         print(type(net))
#         print(net)
#
#         print('logits')
#         print(logits)
#     #input('Press ENTER to continue.')