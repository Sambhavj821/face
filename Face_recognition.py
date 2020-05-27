from keras.applications import VGG16
import numpy as np


# In[2]:


# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows = 224
img_cols = 224 

#include_top=False removes the output layer of the model
base_model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))


# In[3]:


#To get only the name
base_model.layers[0].__class__.__name__

#To see the input/output of a particular layer
base_model.layers[0].input

#Freezing all the layers by making their trainable=False
for layer in base_model.layers:
    layer.trainable=False
    
#Checking this
base_model.layers[10].trainable


# In[4]:


# Let's print our layers 
for (i,layer) in enumerate(base_model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[5]:


#Ouput of the current model
base_model.output


# In[6]:


#Now we add our dense layers for a new prediction on top of the base model
from keras.layers import Dense, Flatten
from keras.models import Sequential

top_model = base_model.output
top_model = Flatten()(top_model)
top_model = Dense(512, activation='relu')(top_model)   #First added FCL dense layer
top_model = Dense(512, activation='relu')(top_model)    #Second added FCL dense layer
top_model = Dense(256, activation='relu')(top_model)    #Third added FCL dense layer
top_model = Dense(4, activation='softmax')(top_model)    #Output layer with 2 class labels


# In[7]:


#Now let's see the top_model output
top_model


# In[8]:


base_model.input


# In[9]:


#IMP: Mounting the base_model with the top_model and forming newmodel

from keras.models import Model
newmodel = Model(inputs=base_model.input, outputs=top_model)


# In[10]:


newmodel.output


# In[11]:


newmodel.layers


# In[12]:


newmodel.summary()


# In[13]:


#Importing our images for recognition
from keras.preprocessing.image import ImageDataGenerator

train_data="\\mlops1/train/"
test_data="\\mlops1/test/"

#Data image augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(img_rows, img_cols),
        class_mode='categorical')

'''#Converting to 4D
for img_train in train_generator:
    img_train=np.expand_dims(img_train, axis=0)'''
 
test_generator = test_datagen.flow_from_directory(
        test_data,
        target_size=(img_rows, img_cols),
        class_mode='categorical',
        shuffle=False)

'''#Converting to 4D
for img_test in test_generator:
    img_test=np.expand_dims(img_test, axis=0)'''


# In[14]:


#Now let's compile our model
from keras.optimizers import RMSprop
newmodel.compile(optimizer = RMSprop(lr=0.0001),
                 loss = 'categorical_crossentropy',
                 metrics =['accuracy']
                )


# In[15]:


history = newmodel.fit_generator(train_generator, epochs=5,steps_per_epoch=100, validation_data=test_generator)


# In[28]:


newmodel.save('FaceRecog_VGG16.h4')


# In[29]:


from keras.models import load_model

classifier = load_model('FaceRecog_VGG16.h4')


# In[30]:


#Testing the Model
get_ipython().run_line_magic('matplotlib', 'inline')
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
from matplotlib import pyplot as plt

import os
import cv2
from os import listdir
from os.path import isfile, join

four_person_dict = {"[0]": "navika", 
                      "[1]": "pavni",
                      "[2]": "raghav",
                      "[3]": "sambhav",
                     }

four_person_dict_n = {"navika": "navika", 
                      "pavni": "pavni",
                      "raghav": "raghav",
                      "sambhav": "sambhav",
                      }

def draw_test(name, pred, im):
    person =four_person_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, person, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + four_person_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("\\Users/DELL/Desktop/Face/test/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()


accuracy_file = open('/mlops2/accuracy.txt','w')
accuracy_file.write(str(scores[1]))
accuracy_file.close()

display_matter = open('/mlops2/display_matter.html','r+')
display_matter.read()
display_matter.write('<pre>\n---------------------------------------------\n')
display_matter.write(this_layer)
display_matter.write('\nAccuracy achieved : ' + str(scores[1])+'\n</pre>')
display_matter.close()




