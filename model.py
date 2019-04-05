import csv
from scipy import ndimage
import numpy as np
import cv2

lines = []
with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for line in reader:
            if(i==0):
                i = i+1
            else:
#                 print(line)
                lines.append(line)
            
images = []
measurements = []

for line in lines:
    
    source_path = line[0]
    #print(source_path)
    filename = source_path.split('/')[-1]
    #print(filename)
    current_path = 'data/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5 , input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')