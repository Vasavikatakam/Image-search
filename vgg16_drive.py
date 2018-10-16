from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.optimizers import SGD

#Load the VGG model
image_size=224
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
'''for layer in vgg_conv.layers[:-4]:
    layer.trainable = False'''


 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())

'''model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(num_classes, activation='softmax'))'''

#model.add(layers.Dense())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))



 
# Show a summary of the model. Check the number of trainable parameters

Adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print('vasavi')


#changes



model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])






X_train=np.load("/content/drive/My Drive/GpuNotebooks/trainingsetdrive_20.npy")
Y_train=np.load("/content/drive/My Drive/GpuNotebooks/trainingsetlabelsdrive_20.npy")

X_valid=np.load("/content/drive/My Drive/GpuNotebooks/validationsetdrive_20.npy")
Y_valid=np.load("/content/drive/My Drive/GpuNotebooks/validationsetlabelsdrive_20.npy")

'''X_test=np.load("/content/drive/My Drive/GpuNotebooks/testsetdrive_20.npy")
Y_test=np.load("/content/drive/My Drive/GpuNotebooks/testsetlabelsdrive_20.npy")'''





'''model_json = model.to_json()
with open("/content/drive/My Drive/GpuNotebooks/model1.json", "w") as json_file:
    json_file.write(model_json)'''



'''filepath="/content/drive/My Drive/GpuNotebooks/model1_weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]'''







model.fit(X_train/255.0, Y_train, epochs=40, batch_size=64,validation_data=(X_valid/255.0,Y_valid))




# serialize model to JSON
model_json = model.to_json()
with open("/content/drive/My Drive/GpuNotebooks/model1final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/content/drive/My Drive/GpuNotebooks/model1_weights_bestfinal.h5")
print("Saved model to disk") 









'''scores = model.evaluate(X_test/255.0,Y_test, verbose=0)

print('Test loss:', scores[0])
print('Testaccuracy:', scores[1])'''



img_path = "/content/drive/My Drive/GpuNotebooks/cellphones.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
#img_data=img_data/255.0
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_category = model.predict(img_data)
k=np.argmax(vgg16_category,axis=1)
#print vgg16_category.shape
print k


#print np.argmax(out)











