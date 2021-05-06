import keras
import numpy as np
import sys
import tensorflow as tf
import cv2


def random_crop(x,dn):
    # print(x.shape)
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    # print(out)
    out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)
    return out

def random_crop_black(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def random_crop_white(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0+255
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def augment_data(images):
    for i in range(0,images.shape[0]):
        
        rand_r = np.random.random()
        if  rand_r < 0.25:
            dn = np.random.randint(15,size=1)[0]+1
            images[i] = random_crop(images[i],dn)

        elif rand_r >= 0.25 and rand_r < 0.5:
            dn = np.random.randint(15,size=1)[0]+1
            images[i] = random_crop_black(images[i],dn)

        elif rand_r >= 0.5 and rand_r < 0.75:
            dn = np.random.randint(15,size=1)[0]+1
            images[i] = random_crop_white(images[i],dn)

        
        if np.random.random() > 0.3:
            images[i] = tf.keras.preprocessing.image.random_zoom(images[i], [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        
    return images

def augment_img(image):
        
    rand_r = np.random.random()
    if  rand_r < 0.25:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop(image,dn)

    elif rand_r >= 0.25 and rand_r < 0.5:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop_black(image,dn)

    elif rand_r >= 0.5 and rand_r < 0.75:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop_white(image,dn)

    
    if np.random.random() > 0.3:
        image = tf.keras.preprocessing.image.random_zoom(image, [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        
    return image


def data_generator_reg(X,Y,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p,q = [],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),np.array(q)
                p,q = [],[]
        if p:
            yield augment_data(np.array(p)),np.array(q)
            p,q = [],[]

def data_generator_pose(X,Y,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        if str(type(X)) == "<class 'h5py._hl.dataset.Dataset'>":
            X = np.array([X[i] for i in idxs])
            Y = np.array([Y[i] for i in idxs])
        else:
            X = X[idxs]
            Y = Y[idxs]

        # X = X[idxs]
        # Y = Y[idxs]

        # print(X.shape)
        # print(Y.shape)
        p,q = [],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            if len(p) == batch_size:
                # print(np.array(p).shape)
                yield augment_data(np.array(p)),np.array(q)
                p,q = [],[]
        if p:
            yield augment_data(np.array(p)),np.array(q)
            p,q = [],[]

def data_generator_pose_custom(X,Y,batch_size):
    while True:
        idxs_1 = np.random.permutation(len(X))
        idxs_2 = np.random.permutation(len(X))
        # print(f"[INFO] Subtract 2 random idxs: {np.sum(np.abs(idxs_1 - idxs_2))}")
        # X_1, Y_1 = X[idxs_1], Y[idxs_1]
        # X_2, Y_2 = X[idxs_2], Y[idxs_2]

        X_1, Y_1 = X[0::2], Y[0::2]
        X_2, Y_2 = X[1::2], Y[1::2]

        batch_imgs_1, batch_imgs_2, batch_labels = [], [], []
        for i in range(min(len(X_2), len(X_1))):
            batch_imgs_1.append(X_1[i])
            batch_imgs_2.append(X_2[i])
            # print("[INFO] Shape imgs: ", batch_imgs_1.shape)
            batch_labels.append(np.concatenate((Y_1[i], Y_2[i])))
           
            # print("[INFO] Shape batch imgs: ", augment_data(np.array(batch_imgs_1)).shape)
            # print("[INFO] Shape batch labels: ", np.array(batch_labels).shape)

            if len(batch_imgs_1) == batch_size:
                yield [augment_data(np.array(batch_imgs_1)), augment_data(np.array(batch_imgs_2))], [np.array(batch_labels)]
                batch_imgs_1, batch_imgs_2, batch_labels = [], [], []

        if batch_imgs_1:
            yield [augment_data(np.array(batch_imgs_1)), augment_data(np.array(batch_imgs_2))], [np.array(batch_labels)]
            batch_imgs_1, batch_imgs_2, batch_labels = [], [], []


def data_generator_pose_pure(X,Y,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p,q = [],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            if len(p) == batch_size:
                yield (np.array(p)),np.array(q)
                p,q = [],[]
        if p:
            yield (np.array(p)),np.array(q)
            p,q = [],[]


def data_generator_pose_ms2(X,Y,batch_size):

    Y1 = Y[0]
    Y2 = Y[1]
    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        p,q1,q2 = [],[],[]
        for i in range(len(X)):
            p.append(X[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),[np.array(q1),np.array(q2)]
                p,q1,q2 = [],[],[]
        if p:
            yield augment_data(np.array(p)),[np.array(q1),np.array(q2)]
            p,q1,q2 = [],[],[]


def data_generator_pose_ms3(X,Y,batch_size):

    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        Y3 = Y3[idxs]
        p,q1,q2,q3 = [],[],[],[]
        for i in range(len(X)):
            p.append(X[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            q3.append(Y3[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),[np.array(q1),np.array(q2),np.array(q3)]
                p,q1,q2,q3 = [],[],[],[]
        if p:
            yield augment_data(np.array(p)),[np.array(q1),np.array(q2),np.array(q3)]
            p,q1,q2,q3 = [],[],[],[]

def data_generator_reg_fft_hr(X,Y,batch_size):

    X1 = X[0]
    X2 = X[1]
    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    while True:
        idxs = np.random.permutation(len(X1))
        X1 = X1[idxs]
        X2 = X2[idxs]
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        Y3 = Y3[idxs]
        p1,p2,q1,q2, q3 = [],[],[],[],[]
        for i in range(len(X1)):
            p1.append(X1[i])
            p2.append(X2[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            q3.append(Y3[i])
            if len(p1) == batch_size:
                yield [augment_data(np.array(p1)),augment_data(np.array(p2))],[np.array(q1),np.array(q2),np.array(q3)]
                p1,p2,q1,q2, q3 = [],[],[],[],[]
        if p1:
            yield [augment_data(np.array(p1)),augment_data(np.array(p2))],[np.array(q1),np.array(q2),np.array(q3)]
            p1,p2,q1,q2, q3 = [],[],[],[],[]



def data_generator_reg_pair(X,Y,batch_size):

    X1 = X[0]
    X2 = X[1]
    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    while True:
        idxs = np.random.permutation(len(X1))
        X1 = X1[idxs]
        X2 = X2[idxs]
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        Y3 = Y3[idxs]
        
        p1,p2,q1,q2, q3 = [],[],[],[],[]
        for i in range(len(X1)):
            p1.append(X1[i])
            p2.append(X2[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            q3.append(Y3[i])
            
            if len(p1) == batch_size:
                yield [augment_data(np.array(p1)),np.array(p2)],[np.array(q1),np.array(q2),np.array(q3)]
                p1,p2,q1,q2,q3 = [],[],[],[],[]
        if p1:
            yield [augment_data(np.array(p1)),np.array(p2)],[np.array(q1),np.array(q2),np.array(q3)]
            p1,p2,q1,q2,q3 = [],[],[],[],[]



def data_generator_dex(X,Y,batch_size):

    Y1 = Y[0]
    Y2 = Y[1]

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        p,q1,q2 = [],[],[]
        for i in range(len(X)):
            p.append(X[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),[np.array(q1),np.array(q2)]
                p,q1,q2 = [],[],[]
        if p:
            yield augment_data(np.array(p)),[np.array(q1),np.array(q2)]
            p,q1,q2 = [],[],[]

def data_generator_dex_centerloss(X,Y,batch_size):
    X1 = X[0]
    X2 = X[1]
    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    while True:
        idxs = np.random.permutation(len(X1))
        X1 = X1[idxs] #images
        X2 = X2[idxs] #labels for center loss
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        Y3 = Y3[idxs]
        p1,p2,q1,q2,q3 = [],[],[],[],[]
        for i in range(len(X1)):
            p1.append(X1[i])
            p2.append(X2[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            q3.append(Y3[i])
            if len(p1) == batch_size:
                yield [augment_data(np.array(p1)),np.array(p2)],[np.array(q1),np.array(q2),np.array(q3)]
                p1,p2,q1,q2,q3 = [],[],[],[],[]
        if p1:
            yield [augment_data(np.array(p1)),np.array(p2)],[np.array(q1),np.array(q2),np.array(q3)]
            p1,p2,q1,q2,q3 = [],[],[],[],[]
