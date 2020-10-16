#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=
# GENERATING SAMPLE DATA INFORMATION WITH IMAGE LOCATION AND CORRESPONDING STEERING

# Importing Relavent Libraries
import csv

samples = [] 
correction = 0.6

# Circuit 1 Data      
with open('/home/workspace/Train_sample/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[0]=='center'):
            continue
        samples.append([ '/home/workspace/Train_sample/' + line[0], float(line[3]) ])
        samples.append([ 'flip/home/workspace/Train_sample/' + line[0], -1*float(line[3]) ])
        if (len(line[1])!=0):
            samples.append([ '/home/workspace/Train_sample/' + line[1].split(' ')[1], float(line[3])+correction ])
        if (len(line[2])!=0):
            samples.append([ '/home/workspace/Train_sample/' + line[2].split(' ')[1], float(line[3])-correction ])
            
# Circuit 1 Reverse Data
with open('/home/workspace/Train_data_Circuit1_Reverse_slow/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[0]=='center'):
            continue
        samples.append(['/home/workspace' + (line[0].split('/root/Desktop')[1]), float(line[3])])
        samples.append(['flip/home/workspace' + (line[0].split('/root/Desktop')[1]), -1*float(line[3])])
        if (len(line[1])!=0):
            samples.append([ '/home/workspace' + (line[1].split('/root/Desktop')[1]), float(line[3])+correction ])
        if (len(line[2])!=0):
            samples.append([ '/home/workspace' + (line[2].split('/root/Desktop')[1]), float(line[3])-correction ])     
            
# Circuit 1 another lap slow
with open('/home/workspace/Train_data_Circuit1_slow/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[0]=='center'):
            continue
        samples.append(['/home/workspace' + (line[0].split('/root/Desktop')[1]), float(line[3])])
        samples.append(['flip/home/workspace' + (line[0].split('/root/Desktop')[1]), -1*float(line[3])])
        if (len(line[1])!=0):
            samples.append([ '/home/workspace' + (line[1].split('/root/Desktop')[1]), float(line[3])+correction ])
        if (len(line[2])!=0):
            samples.append([ '/home/workspace' + (line[2].split('/root/Desktop')[1]), float(line[3])-correction ])  
            
# # Circuit 1 noncenter driving
# with open('/home/workspace/Train_data_Circuit1_noncenter/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         if (line[0]=='center'):
#             continue
#         samples.append(['/home/workspace' + (line[0].split('/root/Desktop')[1]), float(line[3])])
#         samples.append(['flip/home/workspace' + (line[0].split('/root/Desktop')[1]), -1*float(line[3])])
#         if (len(line[1])!=0):
#             samples.append([ '/home/workspace' + (line[1].split('/root/Desktop')[1]), float(line[3])+correction ])
#         if (len(line[2])!=0):
#             samples.append([ '/home/workspace' + (line[2].split('/root/Desktop')[1]), float(line[3])-correction ])                 

# Circuit 2 Data
with open('/home/workspace/Train_data_Circuit2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[0]=='center'):
            continue
        samples.append(['/home/workspace' + (line[0].split('/root/Desktop')[1]), float(line[3])])
        samples.append(['flip/home/workspace' + (line[0].split('/root/Desktop')[1]), -1*float(line[3])])
        if (len(line[1])!=0):
            samples.append([ '/home/workspace' + (line[1].split('/root/Desktop')[1]), float(line[3])+correction ])
        if (len(line[2])!=0):
            samples.append([ '/home/workspace' + (line[2].split('/root/Desktop')[1]), float(line[3])-correction ])

# Circuit 2 Reverse Data
with open('/home/workspace/Train_data_Circuit2_Reverse/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[0]=='center'):
            continue
        samples.append(['/home/workspace' + (line[0].split('/root/Desktop')[1]), float(line[3])])
        samples.append(['flip/home/workspace' + (line[0].split('/root/Desktop')[1]), -1*float(line[3])])
        if (len(line[1])!=0):
            samples.append([ '/home/workspace' + (line[1].split('/root/Desktop')[1]), float(line[3])+correction ])
        if (len(line[2])!=0):
            samples.append([ '/home/workspace' + (line[2].split('/root/Desktop')[1]), float(line[3])-correction ])

# Recenter Data
with open('/home/workspace/Train_data_recenter/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[0]=='center'):
            continue
        samples.append(['/home/workspace' + (line[0].split('/root/Desktop')[1]), float(line[3])])
        samples.append(['flip/home/workspace' + (line[0].split('/root/Desktop')[1]), -1*float(line[3])])
        if (len(line[1])!=0):
            samples.append([ '/home/workspace' + (line[1].split('/root/Desktop')[1]), float(line[3])+correction ])
        if (len(line[2])!=0):
            samples.append([ '/home/workspace' + (line[2].split('/root/Desktop')[1]), float(line[3])-correction ])
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=
# GENERATOR 

# Importing Relavent Libraries
import cv2
import numpy as np
import sklearn
import random

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                
                if (batch_sample[0].startswith('flip')): 
                    img = cv2.imread(batch_sample[0].split('flip')[1])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.flip(img, 1)
                    images.append(img)
                    steering.append(batch_sample[1])
                else:
                    img = cv2.imread(batch_sample[0])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    images.append(img)
                    steering.append(batch_sample[1])

            
            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set batch size
batch_size=500

# Compile and train the model using the Generator 
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=
# Neural Network

# Importing Relavent Libraries
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import math

model = Sequential() 
# Preprocessing using Lambda Layer
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))

# NVIDIA 
model.add(Conv2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,1,1,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator,       validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=1, verbose=1)


# Plotting the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_function_withFlip.png')

model.save('model.h5')