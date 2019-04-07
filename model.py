import csv
from scipy import ndimage
import numpy as np
import cv2

#Reading data from CSV
lines = []
with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for line in reader:
            if(i==0):
                i = i+1
            else:
                lines.append(line)
            
images = []
measurements = []
#Image and angle are extracted and adjusted below with (+/-)0.2 factor
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        if(i == 0):
            measurement = float(line[3])
        elif(i == 1):
            measurement = float(line[3]) + 0.2
        elif(i == 2):
            measurement = float(line[3]) - 0.2 
        
        measurements.append(measurement)

#Augmentaion of images is done using code below (Flipping and inverting angle)
augmented_images,augmented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5,verbose = 1)

model.save('model.h5')