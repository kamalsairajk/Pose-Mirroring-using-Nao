# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:26:40 2019

@author: root
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:20:38 2019

@author: root
"""

from keras.preprocessing.image import  img_to_array
##from keras import backend as K
##K.set_image_dim_ordering('th')
from keras.models import load_model
import numpy as np

# Image manipulations and arranging data
import os
from PIL import Image
import theano
theano.config.optimizer="None"
#Sklearn to modify the data
os.chdir("C:\\Users\\root\\Desktop\\1\\predict")
path2="images"
model = load_model('mymodel.h5')
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
files=os.listdir(path2);
img=files[0] 
im = Image.open(path2 + '\\'+img);
imrs = im.resize((50,50))
imrs=img_to_array(imrs);
imrs=imrs.reshape(50,50,3);


x=[]
x.append(imrs)
x=np.array(x);
predictions = model.predict(x)
prediction=predictions.tolist()
l=prediction[0].index(max(prediction[0]))
predictpose=""
if l==0:
    predictpose="foldinghands"
elif l==1:
    predictpose="highhands"
elif l==2:    
    predictpose="namastey"
elif l==3:   
    predictpose="raisehand"
elif l==4:   
    predictpose="salute"
print(predictpose)
f= open("C:\\Users\\root\\Desktop\\1\\robotpose.txt","r+")
f.write(predictpose)