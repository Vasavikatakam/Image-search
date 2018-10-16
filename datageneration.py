import os
import cv2
import numpy as np
from random import shuffle
import gzip
import random
import urllib
from keras.utils import np_utils



def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image


path='/content/drive/My Drive/GpuNotebooks/Allproducts'
categories={}
inputdata=[]
outputdata=[]
folders=os.listdir(path)
count1=0
for value,folder in enumerate(folders):
	if(count1<=19):
		
		g = gzip.open(path+'/'+folder, 'r')
	
		subinputdata=[]
		suboutputdata=[]
		count=0	
		for l in g:
			if(count==0):
				categories[value]=folder
						
			k=eval(l)
		
			key='imUrl'
			if key in k.keys():
				
				try:
					#print(k['imUrl'])
					img=url_to_image(k['imUrl'])
					#cv2.imshow("image", img)
					#cv2.waitKey(0)
					#print("original size")
					#print(img.shape)	
					resized_img=cv2.resize(img,(224,224))
					subinputdata.append(resized_img)
					suboutputdata.append((value))
					count=count+1
					
				except:
					continue
			if(count==600):
				break
		print('completed')
		if(len(subinputdata)>=500):	
			subinputdata = random.sample(subinputdata, 500)
			suboutputdata=random.sample(suboutputdata,500)
			'''for i in range(0,len(subinputdata)):
				cv2.imshow(str(suboutputdata[i]),subinputdata[i])
				cv2.waitKey(0)'''
			for i in range(0,len(subinputdata)):
				inputdata.append(subinputdata[i])
				outputdata.append(suboutputdata[i])
		count1=count1+1


for key,val in categories.items():
    print key, "=>", val


'''inputdata=np.asarray(inputdata,dtype=np.float32).flatten()
outputdata=np.asarray(outputdata,dtype=np.float32).flatten()
inputdata=inputdata.tolist()
outputdata=outputdata.tolist()'''

print(len(inputdata))
print(len(inputdata[0]))


c=list(zip(inputdata,outputdata))
shuffle(c)
inputdata,outputdata=zip(*c)

'''for i in range(0,len(inputdata)):
	cv2.imshow(str(outputdata[i]),inputdata[i])
	cv2.waitKey(0)'''




inputdata=np.asarray(inputdata,dtype=np.float32)


#print(inputdata.shape)
#eval_data=eval_data.reshape(eval_data.shape[0], 3, 64, 64).astype('float32')
#one hot encode outputs
outputdata=np.asarray(outputdata,dtype=np.int32)
#print(outputdata.shape)
outputdata=np_utils.to_categorical(outputdata)



k=inputdata.shape
x1=k[0]*float(7.0/10)
x1=int(x1)
x2=x1+(k[0])*float(2.0/10)
x2=int(x2)
x3=len(inputdata)
print(x1)
print(x2)
print(x3)

trainingset=inputdata[0:x1]
trainingsetlabels=outputdata[0:x1]


np.save('/content/drive/My Drive/GpuNotebooks/trainingsetfinal.npy',trainingset)
np.save('/content/drive/My Drive/GpuNotebooks/trainingsetlabelsfinal.npy',trainingsetlabels)


validationset=inputdata[x1:x2]
validationlabels=outputdata[x1:x2]



np.save('/content/drive/My Drive/GpuNotebooks/validationsetfinal.npy',validationset)
np.save('/content/drive/My Drive/GpuNotebooks/validationsetlabelsfinal.npy',validationlabels)



testset=inputdata[x2:x3]
testsetlabels=outputdata[x2:x3]


np.save('/content/drive/My Drive/GpuNotebooks/testsetfinal.npy',testset)
np.save('/content/drive/My Drive/GpuNotebooks/testsetlabelsfinal.npy',testsetlabels)


print("Input shape:")
print(inputdata.shape)
#print(trainingset)
#print(trainingsetlabels)









































#print(outputdata.shape())

























	
			
	
		



	
