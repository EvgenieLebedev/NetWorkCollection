from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Activation
from tensorflow import keras
import tensorflow as tf

class Neural():

    def jaccard_loss(y_true, y_pred):
        smoothing = 1.
        intersection = tf.reduce_sum(y_true * y_pred, axis = (1, 2))
        union = tf.reduce_sum(y_true + y_pred, axis = (1, 2))
        jaccard = (intersection + smoothing) / (union - intersection + smoothing)
        return 1. - tf.reduce_mean(jaccard)

    def SegNet():
        inp = Input(shape=(256, 256, 3))

        conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
        conv_1_1 = Activation('relu')(conv_1_1)

        conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
        conv_1_2 = Activation('relu')(conv_1_2)

        pool_1 = MaxPooling2D(2)(conv_1_2)


        conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
        conv_2_1 = Activation('relu')(conv_2_1)

        conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
        conv_2_2 = Activation('relu')(conv_2_2)

        pool_2 = MaxPooling2D(2)(conv_2_2)


        conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
        conv_3_1 = Activation('relu')(conv_3_1)

        conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
        conv_3_2 = Activation('relu')(conv_3_2)

        pool_3 = MaxPooling2D(2)(conv_3_2)


        conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
        conv_4_1 = Activation('relu')(conv_4_1)

        conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
        conv_4_2 = Activation('relu')(conv_4_2)

        pool_4 = MaxPooling2D(2)(conv_4_2)

        up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4)
        conc_1 = Concatenate()([conv_4_2, up_1])

        conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1)
        conv_up_1_1 = Activation('relu')(conv_up_1_1)

        conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
        conv_up_1_2 = Activation('relu')(conv_up_1_2)


        up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
        conc_2 = Concatenate()([conv_3_2, up_2])

        conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(conc_2)
        conv_up_2_1 = Activation('relu')(conv_up_2_1)

        conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
        conv_up_2_2 = Activation('relu')(conv_up_2_2)


        up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
        conc_3 = Concatenate()([conv_2_2, up_3])

        conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(conc_3)
        conv_up_3_1 = Activation('relu')(conv_up_3_1)

        conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
        conv_up_3_2 = Activation('relu')(conv_up_3_2)



        up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
        conc_4 = Concatenate()([conv_1_2, up_4])
        conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(conc_4)
        conv_up_4_1 = Activation('relu')(conv_up_4_1)

        conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1)
        result = Activation('sigmoid')(conv_up_4_2) #сигмоид позволяет вернуть данные в виде тепловой карты

        model = Model(inputs=inp, outputs=result)

        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


        model.compile(adam,loss = 'binary_crossentropy', metrics=['accuracy'])

        return(model)

    def Unet():
        inp = Input(shape=(256, 256, 3))

        conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
        conv_1_1 = Activation('relu')(conv_1_1)
        conv_1_1 = Dropout(0.2)(conv_1_1)

        conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
        conv_1_2 = Activation('relu')(conv_1_2)

        pool_1 = MaxPooling2D(2)(conv_1_2)


        conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
        conv2_1 = Dropout(0.2)(conv_2_1)
        conv_2_1 = Activation('relu')(conv_2_1)

        conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
        conv_2_2 = Activation('relu')(conv_2_2)

        pool_2 = MaxPooling2D(2)(conv_2_2)


        conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
        conv_3_1 = Dropout(0.2)(conv_3_1)
        conv_3_1 = Activation('relu')(conv_3_1)

        conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
        conv_3_2 = Activation('relu')(conv_3_2)

        pool_3 = MaxPooling2D(2)(conv_3_2)


        conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
        conv_4_1  = Dropout(0.2)( conv_4_1 )
        conv_4_1 = Activation('relu')(conv_4_1)

        conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
        conv_4_2 = Activation('relu')(conv_4_2)

        pool_4 = MaxPooling2D(2)(conv_4_2)

        up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4)
        conc_1 = Concatenate()([conv_4_2, up_1])

        conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1)
        conv_up_1_1 = Activation('relu')(conv_up_1_1)

        conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
        conv_up_1_2 = Activation('relu')(conv_up_1_2)


        up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
        conc_2 = Concatenate()([conv_3_2, up_2])

        conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(conc_2)
        conv_up_2_1 = Activation('relu')(conv_up_2_1)

        conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
        conv_up_2_2 = Activation('relu')(conv_up_2_2)


        up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
        conc_3 = Concatenate()([conv_2_2, up_3])

        conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(conc_3)
        conv_up_3_1 = Activation('relu')(conv_up_3_1)

        conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
        conv_up_3_2 = Activation('relu')(conv_up_3_2)

        up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
        conc_4 = Concatenate()([conv_1_2, up_4])
        conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(conc_4)
        conv_up_4_1 = Activation('relu')(conv_up_4_1)

        conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1)
        result = Activation('sigmoid')(conv_up_4_2)

        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model = Model(inputs=inp, outputs=result)


        model.compile(adam, loss='binary_crossentropy',metrics=[Neural.jaccard_loss])

        return(model)