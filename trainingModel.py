import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from patchify import patchify
import segmentation_models as sm

scaler = MinMaxScaler()
root_directory = 'data-set'
patch_size = 256

tile_order = ['Tile 1', 'Tile 2']

image_dataset = [] 
image_order = ['landsat_img_01.jpg', 'landsat_img_02.jpg', 'landsat_img_03.jpg', 'landsat_img_04.jpg', 'landsat_img_05.jpg']
for tile in tile_order:
  #print(tile)
  for path, subdirs, files in os.walk(root_directory):
      #print(subdirs)
      dirname = path.split(os.path.sep)[-1]
      if dirname == tile:
        #print(dirname, path, subdirs)
        images = os.listdir(path+"/images")
        for j in image_order:
          for i, image_name in enumerate(images):  
              if image_name==j:
                print(image_name)
                if image_name.endswith(".jpg"):
                  image = cv2.imread(path+"/images/"+image_name, 1)
                  print(path+"/images/"+image_name)
                  SIZE_X = (image.shape[1]//patch_size)*patch_size
                  SIZE_Y = (image.shape[0]//patch_size)*patch_size
                  image = Image.fromarray(image)
                  image = image.crop((0 ,0, SIZE_X, SIZE_Y))
                  image = np.array(image)             
                  print("Now patchifying image:", path+"/images/"+image_name)
                  patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                  for i in range(patches_img.shape[0]):
                      for j in range(patches_img.shape[1]):   
                          single_patch_img = patches_img[i,j,:,:]
                          single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                          single_patch_img = single_patch_img[0]                        
                          image_dataset.append(single_patch_img)

mask_dataset = [] 
mask_order = ['landsat_img_01.png', 'landsat_img_02.png', 'landsat_img_03.png', 'landsat_img_04.png', 'landsat_img_05.png']
for tile in tile_order:
  #print(tile)
  for path, subdirs, files in os.walk(root_directory):
      #print(subdirs)
      dirname = path.split(os.path.sep)[-1]
      if dirname == tile:
        #print(dirname, path, subdirs)
        masks = os.listdir(path+"/masks")
        for j in mask_order:
          for i, mask_name in enumerate(masks):  
            #print(i, mask_name)
            if mask_name==j:
              if mask_name.endswith(".png"):
                mask = cv2.imread(path+"/masks/"+mask_name, 1)
                print(path+"/masks/"+mask_name)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))
                mask = np.array(mask)             
                print("Now patchifying mask:", path+"/masks/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):   
                        single_patch_mask = patches_mask[i,j,:,:]
                        single_patch_mask = single_patch_mask[0]                        
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset =  np.array(mask_dataset)

import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
print(image_number)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Landsat Image Label")
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.title("Ground Truth Label")
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

Building = '#44079C'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 68, 7, 156
print(Building)

Road = '#0DC2E6'.lstrip('#') 
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #13, 194, 230
print(Road)

Other =  '#9A24FB'.lstrip('#') 
Other = np.array(tuple(int(Other[i:i+2], 16) for i in (0, 2, 4))) #154, 36, 251
print(Other)

label = single_patch_mask


def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Road,axis=-1)] = 1
    label_seg [np.all(label==Other,axis=-1)] = 2
    
    label_seg = label_seg[:,:,0]
    
    return label_seg


labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))


import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Landsat Image Label")
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.title("Ground Truth Label")
plt.imshow(labels[image_number][:,:,0])
plt.show()


n_classes = len(np.unique(labels))
from tensorflow.keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)


weights = [0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


from simple_multi_unet_model import multi_unet_model, jacard_coef  

metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()


history1 = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save("nrsc_project_model.hdf5")

history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from keras.models import load_model
model = load_model("nrsc_project_model.hdf5",
                   custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                   'jacard_coef':jacard_coef})

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Ground Truth Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()