# https://www.kaggle.com/vfdev5/data-visualization
from common import *
from dataset.carvana_cars import *
from net.tool import *

def get_model_from_dir():

   # csv_file  = CARVANA_DIR + '/masks_train.csv'  # read all annotations
   #      mask_dir  = CARVANA_DIR + '/annotations/train'  # read all annotations
   #      df  = pd.read_csv(csv_file)
   #      for n in range(10):
   #          shortname = df.values[n][0].replace('.jpg','')
   #          rle_hat   = df.values[n][1]
   #
   #          mask_file = mask_dir + '/' + shortname + '_mask.gif'
   #          mask = PIL.Image.open(mask_file)
   #          mask = np.array(mask)
   #          rle  = run_length_encode(mask)
   #          #im_show('mask', mask*255, resize=0.25)
   #          #cv2.waitKey(0)
   #          match = rle == rle_hat
   #          print('%d match=%s'%(n,match))

    pass

def split_train_valid_list():
    split = 'train_5088'

    # read names
    split_file = CARVANA_DIR +'/split/'+ split
    with open(split_file) as f:
        names = f.readlines()
    names = [x.strip()for x in names]
    num   = len(names)

    shortnames = [x.replace('train/','')[:-3] for x in names]
    ids = list(set(shortnames))

    num_ids = len(ids)
    print(num_ids) #318

    num_valid=48  #(15%)
    num_train=num_ids-num_valid
    print(num_valid,num_train) #48,270

    random.shuffle(ids)

    #make train, valid
    file1 = CARVANA_DIR +'/split/'+ 'train_v0_%d'%(num_train*CARVANA_NUM_VIEWS)
    file2 = CARVANA_DIR +'/split/'+ 'valid_v0_%d'%(num_valid*CARVANA_NUM_VIEWS)
    ids1 = ids[0:num_train]
    ids2 = ids[num_train: ]

    for pair  in [(file1,ids1),(file2,ids2)]:
        file = pair[0]
        ids  = pair[1]

        with open(file,'w') as f:
            for id in ids:
                for v in range(1,CARVANA_NUM_VIEWS+1):
                    f.write('train/%s_%02d\n'%(id,v))
    xx=0
 # if 1:
 #        img_dir  = CARVANA_DIR + '/images/train'
 #        mask_dir = CARVANA_DIR + '/annotations/train'  # read all annotations
 #
 #        save_dir  = CARVANA_DIR + '/others/small_sfm/no_bg_images640x427'
 #        os.makedirs(save_dir,exist_ok=True)
 #
 #
 #        img_list = glob.glob(img_dir + '/0cdf5b5d0ce1_*.jpg')
 #        num_imgs = len(img_list)
 #        for n in range(num_imgs):
 #            img_file = img_list[n]
 #            shortname = img_file.split('/')[-1].replace('.jpg','')
 #
 #            img_file = img_dir + '/%s.jpg'%(shortname)
 #            img = cv2.imread(img_file)
 #
 #            mask_file = mask_dir + '/%s_mask.gif'%(shortname)
 #            mask = PIL.Image.open(mask_file)   #opencv does not read gif
 #            mask = np.array(mask)
 #
 #            img = img*(np.dstack((mask,mask,mask)))
 #
 #            im_show('img',img, resize=0.25)
 #            im_show('mask',mask*255, resize=0.25)
 #
 #            img_small = cv2.resize(img,(640,427))
 #            cv2.imwrite(save_dir + '/%s.jpg'%shortname, img_small)
 #            cv2.waitKey(1)

    pass

def check_lists_overlap():

    split_file = CARVANA_DIR +'/split/'+ 'train_v0_4320'
    with open(split_file) as f:
        names = f.readlines()
    names1 = [x.strip()for x in names]

    split_file = CARVANA_DIR +'/split/'+ 'valid_v0_768'
    with open(split_file) as f:
        names = f.readlines()
    names2 = [x.strip()for x in names]


    r = bool(set(names1) & set(names2))
    print (r)





def run_make_train_ids():
    split = 'train_5088'

    # read names
    split_file = CARVANA_DIR +'/split/'+ split
    with open(split_file) as f:
        names = f.readlines()
    names = [x.strip()for x in names]
    num   = len(names)


    shortnames = [x.replace('<replace>/','')[:-3] for x in names]
    ids = sorted(list(set(shortnames)))

    num_ids = len(ids)
    print(num_ids) #318

    write_list_to_file(ids, CARVANA_DIR +'/train_5088_ids_318.txt')


def run_make_train_backgrounds():


    # read names
    id_file = CARVANA_DIR +'/train_5088_ids_318.txt'
    with open(id_file) as f:
        ids = f.readlines()
    ids = [x.strip()for x in ids]

    num_ids = len(ids)
    print(num_ids) #318

    for n in range(num_ids):
        bgs = np.zeros((16, CARVANA_HEIGHT,CARVANA_WIDTH,3),np.float32)

        for v in range(16):
            img_file = CARVANA_DIR +'/images/train/%s_%02d.jpg'%(ids[n],v+1)
            img = cv2.imread(img_file)

            #debug
            print (img_file)
            im_show('img', img, resize=0.25)
            cv2.waitKey(1)

            bgs[v] = img
        pass
        bg = bgs.reshape(16,-1,3)
        bg = np.median(bg,0)
        bg = bg.reshape(CARVANA_HEIGHT,CARVANA_WIDTH,3)
        bg = bg.astype(np.uint8)

        #bg = bg/16
        cv2.imwrite(CARVANA_DIR +'/backgrounds/train/%s.jpg'%(ids[n]),bg)
        im_show('bg', bg, resize=0.25)
        cv2.waitKey(1)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_make_train_ids()
    run_make_train_backgrounds()

    print('\nsucess!')