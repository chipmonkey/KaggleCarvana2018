from common import *
# common tool for dataset

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


def draw_text(img, text, pt,  fontScale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color, thickness, cv2.LINE_AA)


##http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=20):

    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if gap==1:
        for p in pts:
            cv2.circle(img,p,thickness,color,-1,cv2.LINE_AA)
    else:
        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return zip(a, a)

        for p, q in pairwise(pts):
            cv2.line(img,p, q, color,thickness,cv2.LINE_AA)

def draw_dotted_poly(img, pts, color, thickness=1, gap=20):

    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        draw_dotted_line(img,s,e,color,thickness,gap)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1, gap=3):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    draw_dotted_poly(img, pts, color, thickness, gap)

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



## custom data transform  -----------------------------------

def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image*std + mean
    image = image.astype(dtype=np.uint8)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return image

def tensor_to_label(tensor):
    label = tensor.numpy()*255
    label = label.astype(dtype=np.uint8)
    return label

def tensor_to_prior(tensor):
    prior = tensor.numpy()*255
    prior = prior.astype(dtype=np.uint8)
    return prior


## transform (input is numpy array, read in by cv2)
def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image-mean)/std
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)   ##.float()
    return tensor

def label_to_tensor(label, threshold=0.5):

    label  = (label>threshold).astype(np.float32)
    tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return tensor

def prior_to_tensor(prior):
    tensor = torch.from_numpy(prior).type(torch.FloatTensor)
    return tensor

# transform image -----------------------------
def random_horizontal_flip(image, u=0.5):

    if random.random() < u:
        image = cv2.flip(image,1)
    return image

def fix_crop(image, roi=(0,0,256,256)):
    x0,y0,x1,y1=roi
    image = image[y0:y1,x0:x1,:]
    return image

def fix_resize(image, w, h):
    image = cv2.resize(image,(w,h))
    return image

# transform image and label -----------------------------

def random_horizontal_flipN(images, u=0.5):

    if random.random() < u:
        for n, image in enumerate(images):
            images[n] = cv2.flip(image,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return images


def random_shift_scale_rotateN(images, shift_limit=(-0.0625,0.0625), scale_limit=(1/1.1,1.1),
                               rotate_limit=(-45,45), aspect_limit = (1,1),  borderMode=cv2.BORDER_REFLECT_101 , u=0.5):
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height,width,channel = images[0].shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        aspect = random.uniform(aspect_limit[0],aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        for n, image in enumerate(images):
            images[n] = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return images



##https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
## color augmentation

#brightness, contrast, saturation-------------
#from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

# def to_grayscle(img):
#     blue  = img[:,:,0]
#     green = img[:,:,1]
#     red   = img[:,:,2]
#     grey = 0.299*red + 0.587*green + 0.114*blue
#     return grey

def random_gray(image, u=0.5):
    if random.random() < u:
        coef  = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray  = np.sum(image * coef,axis=2)
        image = np.dstack((gray,gray,gray))
    return image


def random_brightness(image, limit=(-0.3,0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        image = alpha*image
        image = np.clip(image, 0., 1.)
    return image


def random_contrast(image, limit=(-0.3,0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha*image  + gray
        image = np.clip(image,0.,1.)
    return image


def random_saturation(image, limit=(-0.3,0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587,  0.299]]])
        gray = image * coef
        gray = np.sum(gray,axis=2, keepdims=True)
        image  = alpha*image  + (1.0 - alpha)*gray
        image  = np.clip(image,0.,1.)
    return image

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def random_hue(image, hue_limit=(-0.1,0.1), u=0.5):
    if random.random() < u:
        h = int(random.uniform(hue_limit[0], hue_limit[1])*180)
        #print(h)

        image = (image*255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)/255
    return image


def random_mask_hue(image, mask, hue_limit=(-0.1,0.1), u=0.5):
    if random.random() < u:
        dh = int(random.uniform(hue_limit[0], hue_limit[1])*180)
        image = (image*255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(int)
        hsv[:, :, 0] = mask*((hue + dh) % 180) + (1-mask)*hue
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)/255
    return image,label


## sampler  -----------------------------------

class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples


# see trorch/utils/data/sampler.py
class RandomSamplerWithLength(Sampler):
    def __init__(self, data, length):
        self.num_samples = length
        self.len_data= len(data)

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        l = list(range(self.len_data))
        random.shuffle(l)
        l= l[0:self.num_samples]
        return iter(l)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



