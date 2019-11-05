from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
##from keras import backend as K
##K.set_image_dim_ordering('th')

import numpy as np

# Image manipulations and arranging data
import os
from PIL import Image
import theano
theano.config.optimizer="None"
#Sklearn to modify the data

from sklearn.model_selection import train_test_split
os.chdir("C:\\Users\\root\\Desktop\\Pytorch0.4.1_Openpose-master") ###dataset1 contains images and images1 folder 

# input image dimensions
m,n = 50,50

path2="data";###contains test images
####contains 200 folders of each type of bird

def label_img(img): 
    word_label = img.split('.')[0] 
    # DIY One hot encoder 
    if word_label.startswith('f'): return [1, 0,0,0,0] 
    elif word_label.startswith('h'): return [0, 1,0,0,0]
    elif word_label.startswith('n'): return [0,0,1,0, 0]
    elif word_label.startswith('r'): return [0,0,0, 1,0]
    elif word_label.startswith('s'): return [0, 0,0,0,1]
  
x=[]
y=[]

imgfiles=os.listdir(path2);
for img in imgfiles:
            label=label_img(img)
            im=Image.open(path2+'\\'+img);
            im=im.convert(mode='RGB')
            imrs=im.resize((m,n))
            imrs=img_to_array(imrs);
            #imrs=imrs.transpose(2,0,1);
            #imrs=imrs.reshape(3,m,n);
            imrs=imrs.reshape(m,n,3)
            x.append(imrs)
            y.append(label)
        
x=np.array(x);
y=np.array(y);

batch_size=32
nb_classes=5
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)
#
#uniques, id_train=np.unique(y_train,return_inverse=True)
#Y_train=np_utils.to_categorical(id_train,nb_classes)
#uniques, id_test=np.unique(y_test,return_inverse=True)
#Y_test=np_utils.to_categorical(id_test,nb_classes)

model= Sequential()
#model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=(50,50,3)))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=20;
batch_size=5;
model.fit(x_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, y_test))

#####testing part
files=os.listdir(path2);
img=files[0] 
im = Image.open(path2 + '\\'+img);
imrs = im.resize((m,n))
imrs=img_to_array(imrs);
imrs=imrs.reshape(m,n,3);


x=[]
x.append(imrs)
x=np.array(x);
predictions = model.predict(x)

model.save("mymodel.h5")