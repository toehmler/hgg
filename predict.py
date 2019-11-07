from model import model

class PredictionPipeline(object):

    def __init__(self, model_name):
        self.model_name = model_name

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.root = config['root']
        self.model = model.tri_path(input_shape = (240, 240, 4))
        self.model.load_weights('outputs/models/{}.hdf5'.format(model_name))

    def predict_patient(self, patient_no):
        # read / load scans for patient
        # set labels in gt to combine 1+3 and 
        # normalize slices using same scheme from training
        # transform to channels_last
        # perform actual prediction
        # return np.arrays of prediction and gt

    def predict_slice(self, patient_no, slice_no, show=True):

        # load slice 
        # set labels in gt to combine 1 and 3
        # normalize slice
        # transform to channels last
        # perform prediction
        # return np.array of prediction 
        # save image of prediction if show = true

    def evaluate_patient_prediction(self, pred, gt):

        dice_whole = DSC_whole(pred, gt)
        dice_en = DSC_en(pred, gt)
        dice_core = DSC_core(pred, gt)

        sen_whole = sensitivity_whole(pred, gt)
        sen_en = sensitivity_en(pred, gt)
        sen_core = sensitivity_core(pred, gt)

        spec_whole = specificity_whole(pred, gt)
        spec_en = specificity_en(pred, gt)
        spec_core = specificity_core(pred, gt)

        haus_whole = hausdorff_whole(pred, gt)
        haus_en = hausdorff_en(pred, gt)
        haus_core = hausdorff_core(pred, gt)

        print("=======================================")
        print("Dice whole tumor score: {:0.4f}".format(dice_whole)) 
        print("Dice enhancing tumor score: {:0.4f}".format(dice_en)) 
        print("Dice core tumor score: {:0.4f}".format(dice_core)) 
        print("=======================================")
        print("Sensitivity whole tumor score: {:0.4f}".format(sen_whole)) 
        print("Sensitivity enhancing tumor score: {:0.4f}".format(sen_en)) 
        print("Sensitivity core tumor score: {:0.4f}".format(sen_core)) 
        print("=======================================")
        print("Specificity whole tumor score: {:0.4f}".format(spec_whole)) 
        print("Specificity enhancing tumor score: {:0.4f}".format(spec_en)) 
        print("Specificity core tumor score: {:0.4f}".format(spec_core)) 
        print("=======================================")
        print("Hausdorff whole tumor score: {:0.4f}".format(haus_whole)) 
        print("Hausdorff enhancing tumor score: {:0.4f}".format(haus_en)) 
        print("Hausdorff core tumor score: {:0.4f}".format(haus_core)) 
        print("=======================================\n\n")

     def norm_slices(self,slice_not):
        '''
            normalizes each slice, excluding gt
            subtracts mean and div by std dev for each slice
            clips top and bottom one percent of pixel intensities
        '''
        normed_slices = np.zeros((4, 155, 240, 240))
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(155):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        return normed_slices    


    def _normalize(self,slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp==tmp.min()]=-9
            return tmp


    



        





