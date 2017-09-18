from common import *
from dataset.carvana_cars import *



def prob_to_csv(gz_file, names, probs, log=None, threshold=128):

    assert(probs.dtype==np.uint8)
    num_test = len(probs)

    start = timer()
    rles=[]
    for n in range(num_test):
        if (n%1000==0):
            end  = timer()
            time = (end - start) / 60
            time_remain = (num_test-n-1)*time/(n+1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
        #-----------------------------

        prob = probs[n]

        #case one
        prob = cv2.resize(prob,(CARVANA_WIDTH,CARVANA_HEIGHT))
        mask = prob>threshold

        #case two
        # mask = (prob>threshold).astype(np.float32)
        # mask = cv2.resize(mask,(CARVANA_WIDTH,CARVANA_HEIGHT))
        # mask = mask>0.5 #*255

        rle = run_length_encode(mask)
        rles.append(rle)

        if n<5: #for debug
            print(names[n])
            im_show('mask', mask*255, resize=0.333)
            cv2.waitKey(0)


    #fixe corrupted image
    # https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37247
    if 1:
        n = names.index('29bb3ece3180_11.jpg')
        mask = cv2.imgread('/root/share/project/kaggle-carvana-cars/data/others/ave/11.png',cv2.IMREAD_GRAYSCALE)>128
        rle  = run_length_encode(mask)
        rles[n] = rle



    if log is not None:
        log.write('\trle time = %f min\n'%((timer() - start) / 60)) #20 min

    start = timer()
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')
    if log is not None:
        log.write('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #3 min
