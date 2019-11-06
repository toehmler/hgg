from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add
from keras.layers import Maximum
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.regularizers import l1_l2
from losses import *

class model(object):
    
    def __init__(self, input_shape, load_weights=None):
        self.input_shape = input_shape
        self.load_weights = load_weights
        self.tri_path = self.tri_path()

    def tri_path(input_shape):

        # consider adding gaussian nose in first layer to combat overfitting
        # consider switching from relu to PReLU
         
        X_input = Input(input_shape)

        local = Conv2D(64, (4,4),
                strides=(1,1), padding='valid', activation='relu')(X_input)
        local = BatchNormalization()(local)
        local = Conv2D(64, (4,4),
                strides=(1,1), padding='valid', activation='relu')(local)
        local = BatchNormalization()(local)
        local = Conv2D(64, (4,4),
                strides=(1,1), padding='valid', activation='relu')(local)
        local = BatchNormalization()(local)
        local = Conv2D(64, (4,4),
                strides=(1,1), padding='valid', activation='relu')(local)
        local = BatchNormalization()(local)

        inter = Conv2D(64, (7,7),
                strides=(1,1), padding='valid', activation='relu')(X_input)
        inter = BatchNormalization()(inter)
        inter = Conv2D(64, (7,7),
                strides=(1,1), padding='valid', activation='relu')(inter)
        inter = BatchNormalization()(inter)

        uni = Conv2D(160, (13,13),
                strides=(1,1), padding='valid', activation='relu')(X_input)
        uni = BatchNormalization()(uni)

        out = Concatenate()([local, inter, uni])
        out = Conv2D(4,(21,21),strides=(1,1),padding='valid')(out)
        out = Activation('softmax')(out)

        model = Model(inputs=X_input, outputs=out)

        sgd = SGD(lr=0.08, momentum=0.9, decay=5e-6, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[dice_whole_metric,
                                                                  dice_core_metric,
                                                                  dice_en_metric])
        #load weights if set for prediction
#        if self.load_model_weights is not None:
#            model.load_weights(self.load_model_weights)
        return model



