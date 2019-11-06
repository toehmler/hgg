from model import model 
from glob import glob
import json
import os
import numpy as np
import sys
from keras.models import load_model
from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
import random
import keras.backend as K
from losses import *
from keras.utils import plot_model
from patches import patchExtractor
from keras.utils import np_utils


class SGDLearningRateTracker(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.get_value(optimizer.lr)
        decay = K.get_value(optimizer.decay)
        lr=lr/10
        decay=decay*10
        K.set_value(optimizer.lr, lr)
        K.set_value(optimizer.decay, decay)
        print('LR changed to:',lr)
        print('Decay changed to:',decay)

class TrainingPipeline(object):

    def __init__(self, batch_size, epochs, load_model=None):
        
        self.batch_size = batch_size
        self.epochs = epochs

        with open('config.json') as config_file:
            config = json.load(config_file)
        self.root = config['root']

        if load_model is not None:
            self.model = load_model(load_model, 
                         custom_objects={'gen_dice_loss': gen_dice_loss,
                                         'dice_whole_metric':dice_whole_metric,
                                         'dice_core_metric':dice_core_metric,
                                         'dice_en_metric':dice_en_metric})
            print("pre-trained model loaded")
        else: #space to add more variations
            tri_path = model.tri_path(input_shape = (33,33,4))
            self.model = tri_path
            print("Tri-path compiled")


    def save_model(self, name):
        weights = '/outputs/models/{}.hdf5'.format(name)
        self.model.save_weights(weights)
        print('Model weights saved')
    
    def fit_model(self, x_train, y_train, x_valid, y_valid, name):
        checkpointer = ModelCheckpoint(filepath='outputs/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose = 1)
        self.model.fit(x_train, y_train, 
                        epochs = self.epochs,
                        batch_size = self.batch_size,
                        validation_data = (x_valid, y_valid), 
                        verbose = 1,
#                        callbacks = [checkpointer,
#                                    SGDLearningRateTracker()])

    def generate_patches(self, start, end, h, w):
        patches = patchExtractor(start, end, self.root)
        num_patches = 155 * (end - start) * 3
        x, y = patches.sample_random_patches(num_patches, h, w)
        # transform data to channels_last (keras format)
        x = np.transpose(x, (0,2,3,1)).astype(np.float32)

        # combine necrotic and non-enhancing labels
        y[y == 3] = 1
        y[y == 4] = 3


        # turn y into one-hot encoded
        y_shape = y.shape[0]
        y = y.reshape(-1)
        y = np_utils.to_categorical(y).astype(np.uint8)
        y = y.reshape(y_shape, h, w, 4)
        # shuffle the dataset
        shuffle = list(zip(x, y))
        np.random.seed(180)
        np.random.shuffle(shuffle)
        x = np.array([shuffle[i][0] for i in range(len(shuffle))])
        y = np.array([shuffle[i][1] for i in range(len(shuffle))])
        del shuffle
        return x, y

if __name__ == "__main__":

    # load config stuff to get data
    with open('config.json') as config_file:
        config = json.load(config_file)

    root = config['root']
    print("Root filepath: {}".format(root))

    model_name = input('Model name: ')

    pipeline = TrainingPipeline(batch_size = 4 , epochs = 10)

    # model info and architecture
    print(pipeline.model.summary())
    #plot(pipeline.model, to_file='{}.png'.format(model_name), show_shapes=True)


    # generate training and validation data

    train_start = 0
    train_end = 20 

    valid_start = 95
    valid_end = 100

    h = 33
    w = 33

    print("Generating training data...")
    x_train, y_train = pipeline.generate_patches(train_start, train_end, h, w)
    print("Generating validation data...")
    x_valid, y_valid = pipeline.generate_patches(valid_start, valid_end, h, w)

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.uint8)
    x_valid = x_valid.astype(np.float32)
    y_valid = y_valid.astype(np.uint8)
    
    print("x train shape: {}".format(x_train.shape))
    print("y train shape: {}".format(y_train.shape))
    print("x valid shape: {}".format(x_valid.shape))
    print("y valid shape: {}".format(y_valid.shape))


    # fit the model using pipeline

    pipeline.fit_model(x_train, y_train, x_valid, y_valid, model_name)
    pipeline.save_model(model_name)
    









    

        

    




            



