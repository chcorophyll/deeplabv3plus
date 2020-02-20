import cv2
from glob import glob
from datetime import datetime
import numpy as np
import tensorflow as tf
# import tensorflow.keras.backend as K
import keras.backend as K
from model import Deeplabv3
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model
from keras.models import Model
# from tensorflow.keras.layers import Conv2D, Lambda
from keras.layers import Conv2D, Lambda, Activation
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam


K.clear_session()

def data_loader(img_path, mask_path):
    X_path = glob(img_path + "\*.jpg")
    y_path = glob(mask_path + "\*.png")
    print("data length:", len(X_path))
    print("target length:", len(y_path))
    X_train, X_test, y_train, y_test = train_test_split(X_path, y_path, test_size=0.2)
    return X_train, X_test, y_train, y_test

def mask_preprocessor(data):
    mask = np.uint8(data[:, :, 0] > 0)
    inverse_mask = 1 - mask
    return np.dstack((mask, inverse_mask))

def data_generator(data, targets, batch_size=2):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(data), batch_size * (i + 1)))]
               for i in range(len(data) // batch_size)]
    while True:
        for batch in batches:
            imgs = []
            masks = []
            for i in batch:
                img = cv2.imread(data[i])
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                mask = cv2.imread(targets[i])
                print("img:", img.shape)
                print("mask:", mask.shape)
                mask = mask_preprocessor(mask)
                print("mask_postprocessed:", mask.shape)
                imgs.append(img)
                masks.append(mask)
            imgs = np.array(imgs) / 127.5 - 1
            masks = np.array(masks)

            yield imgs, masks


start_time = datetime.now()
img_path = "C:/Users/Administrator/Desktop/deeplab/CelebAMask-HQ/CelebA-HQ-img"
mask_path = "C:/Users/Administrator/Desktop/deeplab/CelebAMask-HQ/CelebAMask-total-mask"
classes = 2
batch_size = 2
X_train, X_test, y_train, y_test = data_loader(img_path, mask_path)

base_model = Deeplabv3(backbone="xception")
transfered_model = Model(inputs=base_model.input, outputs=base_model.get_layer("decoder_conv1_pointwise_activation").output)
for layer in transfered_model.layers:
    layer.trainable = False

x = Conv2D(classes, (1, 1), padding="same", name="logits_semantic")(transfered_model.output)
# size = tf.int_shape(transfered_model.input)
size_before4 = K.int_shape(transfered_model.input)
# x = Lambda(lambda xx: tf.image.resize(xx, size_before4[1:3], method="bilinear", align_corners=True))(x)
x = Lambda(lambda xx: tf.image.resize(xx, size_before4[1:3], method="bilinear", align_corners=True))(x)
print(x.shape)
x = Activation("sigmoid")(x)
model = Model(inputs=transfered_model.input, outputs=x)

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=100)

history = model.fit_generator(generator=data_generator(X_train, y_train),
                              epochs=50, steps_per_epoch=len(X_train) // batch_size,
                              callbacks=[reduce_lr, early_stopping],
                              validation_data=data_generator(X_test, y_test),
                              validation_steps=len(X_test) // batch_size)
end_time = datetime.now()
print("runtime:", (end_time - start_time).seconds)
model.save_weights("face_model_transfered_weights.h5")
model.save("face_model_transfered.h5")


