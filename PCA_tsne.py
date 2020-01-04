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

def pca2(x,y):
	pca=PCA(2)
	projected=pca.fit_transform(x)
	color=color=[ 'r' if(y[i]==0) else 'b'for i in range(projected.shape[0])]
	plt.scatter(projected[:,0],projected[:,1], c=color)
	plt.show()

def pca3(x,y):
	pca=PCA(3)
	projected=pca.fit_transform(x)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	color=[ 'r' if(y[i]==0) else 'b'for i in range(projected.shape[0])]
	ax.scatter(projected[:,0],projected[:,1],projected[:,2],c=color)
	plt.show()

def tsne2(x,y):
	tsne=TSNE(n_components=2)
	projected=tsne.fit_transform(x)
	color=color=[ 'r' if(y[i]==0) else 'b'for i in range(projected.shape[0])]
	plt.scatter(projected[:,0],projected[:,1], c=color)
	plt.show()

def tsne3(x,y):
	tsne=TSNE(n_components=3)
	projected=tsne.fit_transform(x)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	color=[ 'r' if(y[i]==0) else 'b'for i in range(projected.shape[0])]
	ax.scatter(projected[:,0],projected[:,1],projected[:,2],c=color)
	plt.show()


def load_data(path):
	x_data=[]
	for image in os.listdir(path):
		img=Image.open(path+image)
		img=np.array(img)
		gray_image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_image=gray_image.reshape((gray_image.shape[0]*gray_image.shape[1]))
		x_data.append(gray_image)
	return np.array(x_data)

def load_labels(path):
	y_label=[]
	y_label.append(0)
	for file in os.listdir(path):
		print(file)
		y=pd.read_csv(path+file)
		y=np.array(y)
		for i in range(len(y)):
			y_label.append(y[i])
	return np.array(y_label)

x=load_data("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 2/frames2/lec1/")
y=load_labels("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 2/labels2/")
pca3(x,y)

