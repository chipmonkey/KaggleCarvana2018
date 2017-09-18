from common import *
from dataset.mask import *


#CARVANA_DIR = '/root/share/data/kaggle-carvana-cars-2017'
CARVANA_DIR = '/media/ssd/data/kaggle-carvana-cars-2017'
CARVANA_NUM_VIEWS = 16
CARVANA_HEIGHT = 1280
CARVANA_WIDTH  = 1918



#debug and show
def make_results_image(image, label, prob=None, label_color=2, prob_color=1):

    H,W,C = image.shape
    results = np.zeros((H, 3*W, 3),np.uint8)
    p       = np.zeros((H*W, 3),np.uint8)

    l = np.zeros((H*W),np.uint8)
    m = np.zeros((H*W),np.uint8)
    image1= image.copy()
    if prob is not None:
        m = (prob>127).reshape(-1)
        draw_contour(image1, prob, color=(0,255,0), thickness=1)
    if label is not None:
        l = (label>127).reshape(-1)
        draw_contour(image1, label, color=(0,0,255), thickness=1)

    a = (2*l+m)
    miss = np.where(a==2)[0]
    hit  = np.where(a==3)[0]
    fp   = np.where(a==1)[0]
    p[miss] = np.array([0,0,255])
    p[hit]  = np.array([64,64,64])
    p[fp]   = np.array([0,255,0])
    p       = p.reshape(H,W,3)

    results[:,  0:W  ] = image1
    results[:,  W:2*W] = p
    results[:,2*W:3*W] = image  # image * α + mask * β + λ
    return results




#data iterator ----------------------------------------------------------------

class KgCarDataset(Dataset):

    def __init__(self, split, folder, transform=[], mode='train'):
        super(KgCarDataset, self).__init__()

        # read names
        split_file = CARVANA_DIR +'/split/'+ split
        with open(split_file) as f:
            names = f.readlines()
        names = [name.strip()for name in names]
        num   = len(names)

        #meta data
        df = pd.read_csv(CARVANA_DIR +'/metadata.csv')

        #save
        self.df        = df
        self.split     = split
        self.folder    = folder
        self.transform = transform

        self.mode      = mode
        self.names     = names


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def get_image(self,index):
        name   = self.names[index]
        folder = self.folder
        id     = name[:-3]
        view   = int(name[-2:])-1

        img_file = CARVANA_DIR + '/images/%s/%s.jpg'%(folder,name)
        img   = cv2.imread(img_file)
        image = img.astype(np.float32)/255
        return image

    def get_label(self,index):
        name   = self.names[index]
        folder = self.folder

        #mask_file = CARVANA_DIR + '/annotations/%s/%s_mask.png'%(folder,name)
        if 'test' in folder: mask_file = CARVANA_DIR + '/priors/%s/%s.png'%(folder,name)
        else:                mask_file = CARVANA_DIR + '/annotations/%s/%s_mask.png'%(folder,name)

        mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
        label = mask.astype(np.float32)/255
        return label

    def get_train_item(self,index):
        image = self.get_image(index)
        label = self.get_label(index)

        for t in self.transform:
            image,label = t(image,label)
        image = image_to_tensor(image)
        label = label_to_tensor(label)
        return image, label, index

    def get_test_item(self,index):
        image = self.get_image(index)

        for t in self.transform:
            image = t(image)
        image = image_to_tensor(image)
        return image, index


    def __getitem__(self, index):

        if self.mode=='train': return self.get_train_item(index)
        if self.mode=='test':  return self.get_test_item (index)

    def __len__(self):
        return len(self.names)



#-----------------------------------------------------------------------
def check_dataset(dataset, loader, wait=0):

    if dataset.mode=='train':
        for i, (images, labels, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            num = len(images)
            for n in range(num):
                print(indices[n])
                image = images[n]
                label = labels[n]
                image = tensor_to_image(image, std=255)
                label = tensor_to_label(label)

                results = make_results_image(image, label, prob=None)
                im_show('results', results, resize=0.5)

                #im_show('image', image, resize=1)
                #im_show('label', label, resize=1)
                cv2.waitKey(wait)

    if dataset.mode=='test':
        for i, (images, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            num = len(images)
            for n in range(num):
                image = images[n]
                image = tensor_to_image(image, std=255)

                im_show('image', image, resize=0.5)
                cv2.waitKey(wait)



def run_check_dataset():

    dataset = KgCarDataset('train_v0_4320', 'train512x512', #'train',
                                transform=[
                                    #lambda x,y:  random_horizontal_flip2(x,y),
                                    #lambda x,y:  random_shift_scale_rotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(-0,0)),
                                ],
                                mode='train'
                         )

    if 0: #check indexing
        for n in range(100):
            image, label, index = dataset[n]
            image = tensor_to_image(image, std=255)
            label = tensor_to_label(label)

            im_show('image', image, resize=0.5)
            im_show('label', label, resize=0.5)
            cv2.waitKey(0)

    if 1: #check iterator
        #sampler = FixedSampler(dataset, ([4]*100))
        sampler = FixedSampler(dataset,  list(np.arange(64)+0*16))
        #sampler = SequentialSampler(dataset)
        sampler = RandomSamplerWithLength(dataset,20)
        loader  = DataLoader(dataset, batch_size=4, sampler=sampler,  drop_last=False, pin_memory=True)
        for epoch in range(100):
            print('epoch=%d -------------------------'%(epoch))
            check_dataset(dataset, loader)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()

    print('\nsucess!')