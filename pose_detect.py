import cv2
import argparse
from openpose import Openpose, draw_person_pose

# load model
openpose = Openpose(weights_file = "C:\\Users\\root\\Desktop\\1\\models\\posenet.pth", training = False)

# read image
img = cv2.imread("C:\\Users\\root\\Desktop\\1\\test\\rh.jpeg")

# inference
poses, _ = openpose.detect(img,)

# draw and save image
img1 = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
img2=img1-img
print('Saving result into result.png...')
cv2.imwrite('C:/Users/root/Desktop/1/test/exo/result.png', img2)
print('done')

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
os.chdir("C:\\Users\\root\\Desktop\\1\\test")
path2="exo"
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
    predictpose="folding hands"
elif l==1:
    predictpose="high hands"
elif l==2:    
    predictpose="namastey"
elif l==3:   
    predictpose="raise hand"
elif l==4:   
    predictpose="salute"
print(predictpose)

from gtts import gTTS
import subprocess
import os
tts = gTTS(text='                             '+predictpose, lang='en')
tts.save("C:\\Users\\root\\Desktop\\1\\test\\good.mp3")
#os.system("mpg321 good.mp3")

from pygame import mixer  # Load the external library

mixer.init()
mixer.music.load('C:\\Users\\root\\Desktop\\1\\test\\good.mp3')
mixer.music.play()
