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

def Mean_IOU_tensorflow_2(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)

# def image_resize_l(xx, size_before4):
#     return tf.compat.v1.image.resize(xx, size_before4[1:3], method="bilinear", align_corners=True)

start_time = datetime.now()
img_path = "C:/Users/Administrator/Desktop/deeplab/CelebAMask-HQ/CelebA-HQ-img"
mask_path = "C:/Users/Administrator/Desktop/deeplab/CelebAMask-HQ/CelebAMask-total-mask"
classes = 2
batch_size = 2
X_train, X_test, y_train, y_test = data_loader(img_path, mask_path)

# base_model = Deeplabv3(backbone="xception")
# transfered_model = Model(inputs=base_model.input, outputs=base_model.get_layer("decoder_conv1_pointwise_activation").output)
# for layer in transfered_model.layers:
#     layer.trainable = False
#
# x = Conv2D(classes, (1, 1), padding="same", name="logits_semantic")(transfered_model.output)
# # size = tf.int_shape(transfered_model.input)
# size_before4 = K.int_shape(transfered_model.input)
# # x = Lambda(lambda xx: tf.image.resize(xx, size_before4[1:3], method="bilinear", align_corners=True))(x)
# # x = Lambda(lambda xx: tf.image.resize(xx, size_before4[1:3], method="bilinear", align_corners=True))(x)
# x = Lambda(image_resize_l, arguments={"size_before4": size_before4})(x)
# # print(x.shape)
# x = Activation("sigmoid")(x)
# model = Model(inputs=transfered_model.input, outputs=x)
model = Deeplabv3(classes=2, backbone="xception", activation="sigmoid")
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", Mean_IOU_tensorflow_2])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=100)

history = model.fit_generator(generator=data_generator(X_train, y_train),
                              epochs=50, steps_per_epoch=len(X_train) // batch_size,
                              callbacks=[reduce_lr, early_stopping],
                              validation_data=data_generator(X_test, y_test),
                              validation_steps=len(X_test) // batch_size)
end_time = datetime.now()
print("runtime:", (end_time - start_time).seconds)
model.save_weights("face_model_weights_with_miou.h5")


