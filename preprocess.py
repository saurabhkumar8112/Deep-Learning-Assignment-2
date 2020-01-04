import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pytesseract
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import accuracy_score

os.chdir("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 2")
'''lol=np.array(Image.open("don't talk.JPG"))
print(lol.shape)

text=pytesseract.image_to_string(Image.open("don't talk.JPG"))
print(text)'''
'''x=[]
def plot_function(path):
	for image in os.listdir(path):
		print(image)
		image=np.array(Image.open("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 2/frames/lec10/"+image))
		gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ret,threshold=cv2.threshold(gray_image,150,255,cv2.THRESH_BINARY_INV)
		x.append(cv2.countNonZero(threshold))
		#print(cv2.countNonZero(threshold))
plot_function("frames/lec10")
plt.plot(x)
plt.ylabel("Pixel Density")
plt.xlabel("Frame Number")
plt.show()'''
x=[]
y=[]
def read_labels(path):
	for file in os.listdir(path):
		print(path+file)
		y_label=pd.read_csv(path+file)
		y_label=np.array(np.array(y_label))
		y.append(0)
		for i in range(len(y_label)):
			y.append(y_label[i])
def OCR(path):
	for folder in os.listdir(path):
		print(path+folder)
		for image in os.listdir(path+folder):
			text=pytesseract.image_to_string(path+folder+"/"+image)
			x.append(len(text))
read_labels("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 2/labels2/")
OCR("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 2/frames2/")

y_predict=[]
for i in range(len(x)-1):
	if(i==0):
		y_predict.append(0)
		continue
	else: 
		if((x[i]==0 and x[i+1]<=5) or (x[i]==0 and x[i+1]>5 )):
			y_predict.append(1)
		else:
			y_predict.append(0)
y_predict.append(0)
print(len(y_predict))
print(len(y))
print(len(x))
print(f1_score(y,y_predict))
print(accuracy_score(y,y_predict))
average_precision = average_precision_score(y, y_predict)

precision, recall, _ = precision_recall_curve(y, y_predict)
precision*=5
recall*=5
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.show()




