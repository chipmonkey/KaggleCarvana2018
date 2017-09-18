from common import *
from dataset.carvana_cars import *
from dataset.tool import *

from net.tool import *
from net.rate import *

#post processing
def do_post_process1(prob):


    ret, img = cv2.threshold(prob,128,255,0)#128
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #prob = ((prob>127)*255).astype(np.uint8)


    moments    = [cv2.moments(c) for c in contours]
    areas      = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    indices    = sorted( range(len(contours)), key=lambda i : areas[i],reverse=True )

    x,y,w,h = cv2.boundingRect(contours[indices[0]])
    x0,y0,x1,y1 =x,y,x+w,y+h
    #cv2.rectangle(results,(x0,y0),(x1,y1),(0,255,255),2)

    border0 = int(0.08*h)
    border1 = int(0.50*h)
    ly0, ly1 = y0+border0,y1-border1
    #cv2.line(results,(0,ly0),(W,ly0),(0,0,255),1)
    #cv2.line(results,(0,ly1),(W,ly1),(0,0,255),1)


    for i in indices:
        cnt       = contours[i]
        area      = areas[i]
        perimeter = perimeters[i]
        mt        = moments[i]
        xx,yy,ww,hh = cv2.boundingRect(cnt)
        xx0,yy0,xx1,yy1 = xx,yy,xx+ww,yy+hh

        #center
        cx = int(mt['m10']/(mt['m00']+0.1))
        cy = int(mt['m01']/(mt['m00']+0.1))

        #cv2.drawContours(results, [cnt], -1, (0,255,0), -1)
        if area<1000000 and yy0>ly0 and yy1<ly1:
            cv2.drawContours(prob, [cnt], -1, (255),-1)
        else:
            #cv2.drawContours(results, [cnt], -1, (0,255,0), 3)
            pass

    return prob




# http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html


def run_post_1():

    #img_dir='/media/ssd/data/kaggle-carvana-cars-2017/work/problem00/priors'
    img_dir='/media/ssd/data/kaggle-carvana-cars-2017/priors/test1024x1024'
    img_list = sorted(glob.glob(img_dir + '/*02.png'))
    #img_list = sorted(glob.glob(img_dir + '/*.png'))

    num_imgs = len(img_list)
    for n in range(num_imgs):
        print('n/num_imgs=%d/%d'%(n,num_imgs))

        img_file = img_list[n]
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        prob = do_post_process1(img)


        H,W = img.shape
        ret, img = cv2.threshold(img,127,255,0)#128
        results = np.dstack((img,img,img))

        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        moments    = [cv2.moments(c) for c in contours]
        areas      = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        indices    = sorted( range(len(contours)), key=lambda i : areas[i],reverse=True )

        x,y,w,h = cv2.boundingRect(contours[indices[0]])
        x0,y0,x1,y1 =x,y,x+w,y+h
        cv2.rectangle(results,(x0,y0),(x1,y1),(0,255,255),2)

        border0 = int(0.08*h)
        border1 = int(0.50*h)
        ly0, ly1 = y0+border0,y1-border1
        cv2.line(results,(0,ly0),(W,ly0),(0,0,255),1)
        cv2.line(results,(0,ly1),(W,ly1),(0,0,255),1)

        print(hierarchy)
        for i in indices:

            cnt       = contours[i]
            area      = areas[i]
            perimeter = perimeters[i]
            mt = moments[i]
            print(area)

            xx,yy,ww,hh = cv2.boundingRect(cnt)
            xx0,yy0,xx1,yy1 = xx,yy,xx+ww,yy+hh


            #center
            cx = int(mt['m10']/(mt['m00']+0.1))
            cy = int(mt['m01']/(mt['m00']+0.1))


            #holes
            # https://stackoverflow.com/questions/20492152/how-to-find-no-of-inner-holes-using-cvfindcontours-and-hierarchy
            # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html

            #cv2.drawContours(results, [cnt], -1, (0,255,0), -1)
            if area<1000000 and yy0>ly0 and yy1<ly1:
                cv2.drawContours(results, [cnt], -1, (0,0,255),-1)
            else:
                #cv2.drawContours(results, [cnt], -1, (0,255,0), 3)
                pass


            #draw_shadow_text(results, '%d'%(i), (cx,cy),  1, (255,255,255), 2)
            #
        name = img_file.split('/')[-1].replace('.png','')
        cv2.imwrite('/root/share/project/kaggle-carvana-cars/results/post/xxx'+'/%s.png'%name, results)
        im_show('results',results,0.5)
        im_show('prob',prob,0.5)
        cv2.waitKey(0)

    pass


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_post_1()
    print('\nsucess!')