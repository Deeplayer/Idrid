from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Activation, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications.xception import Xception


def xception():

    base_model = Xception(weights='imagenet', include_top=False, input_shape=(512,512,3))

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    merge_one = Dense(1024, activation='relu', name='fc1')(x)
    merge_one = Dropout(0.5)(merge_one)
    merge_one = Dense(512, activation='relu', name='fc2')(merge_one)
    merge_one = Dropout(0.5)(merge_one)

    predictions = Dense(5, activation='softmax')(merge_one)

    model = Model(input=base_model.input, output=predictions)

    optimizer = Adam(lr=1e-4, decay=0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
