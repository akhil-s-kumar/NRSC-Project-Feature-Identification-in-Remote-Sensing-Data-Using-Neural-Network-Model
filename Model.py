import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from patchify import patchify, unpatchify
import segmentation_models as sm

scaler = MinMaxScaler()
patch_size = 256
n_classes = 3

img = cv2.imread("data-set/Tile 1/images/landsat_img_03.jpg", 1)
original_mask = cv2.imread("data-set/Tile 1/masks/landsat_img_03.png", 1)
original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)

model = load_model("nrsc_project_model.hdf5", compile=False)

SIZE_X = (img.shape[1]//patch_size)*patch_size
SIZE_Y = (img.shape[0]//patch_size)*patch_size
large_img = Image.fromarray(img)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))
large_img = np.array(large_img)     
patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  
patches_img = patches_img[:,:,0,:,:,:]   

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i,j,:,:,:]
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]                     
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                            patches_img.shape[2], patches_img.shape[3]])
unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

plt.figure(figsize=(12, 12))
plt.title('Unpatched Prediction')
plt.imshow(unpatched_prediction)

def label_to_rgb(predicted_image):
    
    Building = '#44079C'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 68, 7, 156

    Road = '#0DC2E6'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) # 13, 194, 230

    Other =  '#9A24FB'.lstrip('#') 
    Other = np.array(tuple(int(Other[i:i+2], 16) for i in (0, 2, 4))) # 154, 36, 251

    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Road
    segmented_img[(predicted_image == 2)] = Other
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

prediction=label_to_rgb(unpatched_prediction)

plt.figure(figsize=(12, 12))
plt.subplot(231)
plt.title('Original Image')
plt.imshow(img)
plt.subplot(232)
plt.title('Ground Truth Image')
plt.imshow(original_mask)
plt.subplot(233)
plt.title('Predicted Image')
plt.imshow(prediction)
plt.show()