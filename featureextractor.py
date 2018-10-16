
import cv2
import math
import numpy as np
from keras.models import model_from_json
from keras.backend import clear_session


def leafcount(imagepath):
	testimage=[]
	img=cv2.imread(imagepath)
	resized_img=cv2.resize(img,(64,64))
	images=np.asarray(resized_img)
	images=images.flatten()
	testimage.append(images)
	testdata=np.asarray(testimage,dtype=np.float32)
	testdata=testdata.reshape(testdata.shape[0], 3, 64, 64).astype('float32')
	# normalize inputs from 0-255 to 0-1
	testdata = testdata/255


	
	json_file = open('/home/vasavi/Downloads/JIvassInternship/updatedGpufiles1/leafcount_convnet.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	#convmodel=loaded_model(weights='/home/vasavi/Downloads/JIvassInternship/updatedGpufiles1/leafcount_convnet.h5',include_top=False)
	# load weights into new model

	loaded_model.load_weights('/home/vasavi/Downloads/JIvassInternship/updatedGpufiles1/leafcount_convnet.h5')
	loaded_model.pop()
	loaded_model.pop()
	loaded_model.pop()
	#loaded_model.pop()
	loaded_model.summary()
	#scores = loaded_model.evaluate(eval_data, eval_labels, verbose=0)
	y_predicted = loaded_model.predict(testdata)
	
	print(y_predicted.shape)
	
	clear_session()

	return y_predicted

k=leafcount('/home/vasavi/Desktop/AllFiles/style.png')
print(k)
