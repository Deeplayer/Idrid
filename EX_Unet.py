from keras.models import Model
from keras.optimizers import Adam, SGD, Adadelta
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, merge, Dropout, Conv2DTranspose
from keras import backend as K


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. / (dice_coef(y_true, y_pred) + smooth)


def unet1(patch_height=128, patch_width=128, n_ch=3):

    inputs = Input(shape=(patch_height, patch_width, n_ch))

    conv1 = Conv2D(64, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(inputs)
    #conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(pool1)
    #conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(conv5)
    conv5 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(conv5)
    #drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(pool5)
    conv6 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(conv6)

    up7 = Conv2D(256, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv5, up7], axis=3)
    conv7 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(merge7)

    up8 = Conv2D(256, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv4, up8], axis=3)
    conv8 = Conv2D(512, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(merge8)

    up9 = Conv2D(128, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv3, up9], axis=3)
    conv9 = Conv2D(256, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(merge9)

    up10 = Conv2D(64, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(UpSampling2D(size=(2, 2))(conv9))
    merge10 = concatenate([conv2, up10], axis=3)
    conv10 = Conv2D(128, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(merge10)

    up11 = Conv2D(32, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(UpSampling2D(size=(2, 2))(conv10))
    merge11 = concatenate([conv1, up11], axis=3)
    conv11 = Conv2D(32, 3, activation='relu', kernel_initializer='lecun_uniform', padding='same')(merge11)

    conv12 = Conv2D(1, 1, activation='sigmoid')(conv11)

    model = Model(inputs=inputs, outputs=conv12)
    opt = Adam(lr=1e-1, decay=0)
    model.compile(optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def unet():
    inputs = Input((128, 128, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])

    return model


# model = unet()
# model.summary()
