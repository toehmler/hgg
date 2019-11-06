'''
patches.py
'''

from glob import glob
import SimpleITK as stik
import numpy as np
import json
from tqdm import tqdm
from keras.utils import np_utils

class patchExtractor(object):
    
    def __init__(self, start, end, root):
        self.root = root
        self.start = start
        self.end = end
        self.training_scans = self.load_training_scans()


    def load_training_scans(self):
        train_scans = []
        i = self.start
        num_patients = self.end - self.start

        # read scans for each patient into size (5,155,240,240)
        for j in tqdm(range(num_patients)):
            flair = glob(self.root + '/*pat{}*/*Flair*/*.mha'.format(i))
            t1 = glob(self.root + '/*pat{}*/*T1.*/*_n4.mha'.format(i))
            t1c = glob(self.root + '/*pat{}*/*T1c.*/*_n4.mha'.format(i))
            t2  = glob(self.root + '/*pat{}*/*T2.*/*.mha'.format(i))
            gt = glob(self.root + '/*pat{}*/*OT*/*.mha'.format(i))
            paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
            scans = [stik.GetArrayFromImage(stik.ReadImage(paths[mod])) 
                    for mod in range(len(paths))]
            scans = np.array(scans)
            scans = self.norm_scans(scans)
            train_scans.append(scans)
            del scans
            i += 1
        return np.array(train_scans)
                
    def norm_scans(self, scans):
        '''
            normalizes each slice , excluding gt
        '''
        # one modality at a time, going slice by slice
        normed_scans = np.zeros((5,155,240,240)).astype(np.float32)
        for mod_idx in range(4):
            normed_scans[mod_idx] = scans[mod_idx]
            for scan_idx in range(155):
                normed_scans[mod_idx][scan_idx] = self.normalize(scans[mod_idx][scan_idx])
        normed_scans[-1] = scans[-1] # gt does not change
        return normed_scans

    def normalize(self, scan):
        '''
            subtracts mean and div by std dev for each slice
            clips top and bottom one percent of pixel intensities
        '''
        bottom = np.percentile(scan, 99)
        top = np.percentile(scan, 1)
        scan = np.clip(scan, top, bottom)
        scan_nonzero = scan[np.nonzero(scan)]
        if np.std(scan) == 0 or np.std(scan_nonzero) == 0:
            return scan
        else:
            # since the range of intensities is between 0 and 5000 ,
            # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
            # the min is replaced with -9 just to keep track of 0 intensities 
            # so that we can discard those intensities afterwards when sampling random patches

            tmp = (scan - np.mean(scan_nonzero)) / np.std(scan_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp
            
    def sample_random_patches(self, num, h, w):
        print("Sampling {} patches randomly.".format(num))

        patches, labels = [], []
        count = 0

        # swap axis 0 -> become modality, 1 -> scan
        gt_scan = np.swapaxes(self.training_scans, 0, 1)[4]
        # use flair as mask
        mask = np.swapaxes(self.training_scans, 0, 1)[0]
        # save shape of gt image
        tmp_gt_shape = gt_scan.shape
        # reshape mask and gt to 1d array
        gt_scan = gt_scan.reshape(-1).astype(np.uint8)
        mask = mask.reshape(-1).astype(np.float32)
        # keep list of 1d indices while discarding 0 intensities 
        indices = np.squeeze(np.argwhere((mask !=-9.0) & (mask != 0.0)))
        del mask

        # shuffle indicies
        np.random.shuffle(indices)
        # restore shape of gt
        gt_scan = gt_scan.reshape(tmp_gt_shape)

        pbar = tqdm(total = num)

        # sample patches from scans
        i = 0
        pix = len(indices)
        while (count < num) and (pix > i):
            # choose a random index
            ind = indices[i]
            i+= 1
            #reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_gt_shape)
            # get the patient and the slice id
            patient_id = ind[0]
            slice_idx = ind[1]
            p = ind[2:]
            #construct the patch by defining the coordinates
            p_y = (p[0] - (h)/2, p[0] + (h)/2)
            p_x = (p[1] - (w)/2, p[1] + (w)/2)
            p_x = list(map(int,p_x))
            p_y = list(map(int,p_y))

            # combine pathches from other modalities
            tmp = self.training_scans[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
            # get label patch
            label = gt_scan[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]

            # only keep patches of the correct size
            if tmp.shape != (4, h, w):
                continue
            patches.append(tmp)
            labels.append(label)
            count += 1
            pbar.update(1)
        patches = np.array(patches)
        labels = np.array(labels)
        pbar.close()
        print("Patch sampling complete")
        return patches, labels

 

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']

    start = input('Start: ')
    end = input('End: ')

    # gives 3 patches per slice
    num_patches = 155*(end-start)*3 

    extractor = patchExtractor(start=start, end=end, root=root)
    patches, labels  = extractor.sample_random_patches(num=num_patches, h=33, w=33)
    # transform data to channels_last (keras format)
    patches = np.transpose(patches, (0,2,3,1)).astype(np.float32)

    # turn y into one-hot encoding for keras 
    label_shape = labels.shape[0]
    labels = labels.reshape(-1)
    labels = np_utils.to_categorical(labels).astype(np.uint8)
    labels = labels.reshape(label_shape, 33, 33, 5)

    # shuffle dataset
    shuffle = list(zip(patches, labels))
    np.random.seed(180)
    np.random.shuffle(shuffle)
    patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    del shuffle

    print("patches shape: {}".format(patches.shape))
    print("labels shape: {}".format(labels.shape))










    print(patches.shape) # (1395, 4, 33, 33)
    print(labels.shape) # (1395, 33, 33)






