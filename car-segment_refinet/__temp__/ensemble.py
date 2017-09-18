from common import *
from submit import *
from dataset.carvana_cars import *
from net.tool import *

def run_vote():

    prediction_files=[
        '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2/submit/probs.8.npy',
        '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2_two-loss/submit/probs.8.npy',
        '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2_two-loss-full_1/submit/probs.8.npy',
    ]
    out_dir ='/root/share/project/kaggle-carvana-cars/results/ensemble/xxx'

    log = Logger()
    log.open(out_dir+'/log.vote.txt',mode='a')
    os.makedirs(out_dir,  exist_ok=True)

    write_list_to_file(prediction_files, out_dir+'/prediction_files.txt')

    #----------------------------------------------------------

    #read names
    split_file = CARVANA_DIR +'/split/'+ 'test%dx%d_100064'%(CARVANA_H,CARVANA_W)
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    names = [name.split('/')[-1]+'.jpg' for name in names]

    #read probs
    num_test   = len(names)
    votes = np.zeros((num_test, CARVANA_H, CARVANA_W), np.uint8)

    num_files = len(prediction_files)
    for n in range(num_files):
        prediction_file = prediction_files[n]
        print(prediction_files[n])

        probs = np.load(prediction_file)
        votes += probs >=128
        probs = None



    #prepare csv file -------------------------------------------------------
    threshold = 1  #/num_files
    probs = votes

    gz_file = out_dir+'/results-ensemble-th%05f.csv.gz'%threshold
    prob_to_csv(gz_file, names, votes, log, threshold)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_vote()

    print('\nsucess!')