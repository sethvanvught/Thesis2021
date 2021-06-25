#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Data preprocessing
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Deep learning
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Conv3D, MaxPooling3D,GlobalAveragePooling3D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf 
from tensorflow.keras.preprocessing import image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
import cv2, os, gc, glob
from tqdm import tqdm

from tensorflow.keras import layers, models

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical, np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# In[7]:


# Image paths

imagePaths = []
for dirname, _, filenames in os.walk('/Users/Seth/Downloads/COVID-19_Radiography_Dataset'):
    for filename in filenames:
        if (filename[-3:] == 'png'):
            imagePaths.append(os.path.join(dirname, filename))


# In[8]:


# Data - Label

Data = []
Target = []
resize = 150

cat = {'Normal': 'Normal', 'COVID': 'Covid-19'}

for imagePath in tqdm(imagePaths):
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image2 = cv2.resize(image, (resize, resize))
    image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image4 = cv2.equalizeHist(image3)
    backtorgb = cv2.cvtColor(image4, cv2.COLOR_GRAY2BGR)


    Data.append(backtorgb)
    Target.append(cat[label])


# In[9]:


# Count by Label

df = pd.DataFrame(Target,columns=['Labels'])
sns.countplot(df['Labels'])
plt.show()

print('Normal:',Target.count('Normal'))
print('Covid-19:',Target.count('Covid-19'))


# In[5]:


#le = LabelEncoder()
#labels = le.fit_transform(Target)
#labels = to_categorical(labels)

#print(le.classes_)
#print(labels[0])


# In[10]:


# encode text category labels 
from sklearn.preprocessing import LabelEncoder 
 
le = LabelEncoder() 
labels = le.fit(Target) 
labels = le.transform(Target) 
 
print(labels[1495:1505])


# In[30]:


plt.imshow(Data[0], cmap='gray')


# In[11]:


(x_train, x_test, y_train, y_test) = train_test_split(Data, labels,test_size=0.20,
                                                      stratify=labels,random_state=42)

trainX = np.array(x_train)
testX = np.array(x_test)
trainY = np.array(y_train)
testY = np.array(y_test)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)


# In[31]:


plt.imshow(trainX[0])


# In[12]:


from keras.applications.resnet50 import preprocess_input
train_datagen = ImageDataGenerator(zoom_range=0.3, rotation_range=50,
 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
 horizontal_flip=True, fill_mode='nearest', preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[13]:


train_generator = train_datagen.flow(trainX, trainY, batch_size=32, shuffle = True)
val_generator = val_datagen.flow(testX, testY, batch_size=32, shuffle = True)


# In[14]:


from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(150,150,3))

output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

restnet = Model(restnet.input, output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()


# In[15]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(150,150,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=2e-5),
              metrics=['accuracy'])
model.summary()


# In[16]:


history = model.fit(train_generator, 
                              steps_per_epoch=20, 
                              epochs=20,
                              validation_data=val_generator, 
                              validation_steps=20, 
                              verbose=1)


# In[17]:


modelLoss, modelAccuracy = model.evaluate(testX, testY, verbose=0)

print('Test Loss is {}'.format(modelLoss))
print('Test Accuracy is {}'.format(modelAccuracy ))


# In[30]:


modelLoss2, modelAccuracy2 = model.evaluate(trainX, trainY, verbose=0)

print('Train Loss is {}'.format(modelLoss2))
print('Train Accuracy is {}'.format(modelAccuracy2 ))


# In[22]:


class_names = ['COVID-19','Normal']

y_pred = model.predict(testX)
print(class_names[np.argmax(y_pred[1])])


# In[25]:


# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)


# In[26]:


# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]


# In[1]:


# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)


# In[27]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testY, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testY, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testY, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testY, yhat_classes)
print('F1 score: %f' % f1)


# In[28]:


from matplotlib import pyplot
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel("Epochs")
x_ticks = np.arange(0, 21, 3)
plt.xticks(x_ticks)
pyplot.legend()


# In[29]:


# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.xlabel("Epochs")
x_ticks = np.arange(0, 21, 3)
plt.xticks(x_ticks)
pyplot.show()


# In[17]:


# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()


# In[ ]:




