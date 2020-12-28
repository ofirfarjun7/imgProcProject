import h5py
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import Dataset

class CharsDataset(Dataset):
	def __init__(self):
		self.db = h5py.File('SynthText.h5', 'r')
		im_names = list(self.db['data'].keys())

		self.samples = []
		for im_name in im_names:
			charBB = db['data'][im_name].attrs['charBB']
			fonts = db['data'][im_name].attrs['font']
			charBBSquares = getSquarePoints(charBB)
    		
			for idx, square in enumerate(charBBSquares):
				charOb = {'image_name': im_name, 'idx': idx, 'square': square[:], 'label': fonts[idx]}
				self.samples.append(charOb)

		self.samples = self.samples[:3]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		charOb = self.samples[idx]
		square = charOb['square']
		image_name = charOb['image_name']

		img = self.db['data'][image_name][:]
		height = img.shape[0]
		width = img.shape[1]

		min_x = int(min([square[0].x, square[1].x, square[2].x, square[3].x]))
		max_x = int(max([square[0].x, square[1].x, square[2].x, square[3].x]))
		min_y = int(min([square[0].y, square[1].y, square[2].y, square[3].y]))
		max_y = int(max([square[0].y, square[1].y, square[2].y, square[3].y]))

		crop_img = img[min_y:max_y, min_x:max_x]
		crop_img = cv2.resize(crop_img, (128, 128))
		crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

		return crop_gray, charOb['label']

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __str__(self):
		return '({0}, {1})'.format(self.x, self.y)

db = h5py.File('SynthText.h5', 'r')
im_names = list(db['data'].keys())

def getSquarePoints(BB):
	num_of_points = len(BB[0][0])
	squares = []

	for idx in range(num_of_points):
		points = []
		for p_idx in range(4):
			points.append(Point(BB[0][p_idx][idx], BB[1][p_idx][idx]))

		squares.append(points)

	return squares

def showImageWordsAndChars(im):
	img = db['data'][im][:]

	height = img.shape[0]
	width = img.shape[1]

	font = db['data'][im].attrs['font']
	txt = db['data'][im].attrs['txt']
	charBB = db['data'][im].attrs['charBB']
	wordBB = db['data'][im].attrs['wordBB']

	# num_of_chars
	charBBSquares = getSquarePoints(charBB)
	
	for square in charBBSquares[:1]:
		
		min_x = int(min([square[0].x, square[1].x, square[2].x, square[3].x]))
		max_x = int(max([square[0].x, square[1].x, square[2].x, square[3].x])) 
		
		min_y = int(min([square[0].y, square[1].y, square[2].y, square[3].y]))
		max_y = int(max([square[0].y, square[1].y, square[2].y, square[3].y]))
		
		crop_img = img[min_y:max_y, min_x:max_x]
		crop_img = cv2.resize(crop_img, (128, 128))
		crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		plt.title('cropped')
		plt.imshow(crop_gray, cmap='gray')
		plt.show()

		# cv2.line(img, (int(square[0].x),int(square[0].y)), (int(square[1].x),int(square[1].y)), (255,0,0), 1)
		# cv2.line(img, (int(square[1].x),int(square[1].y)), (int(square[2].x),int(square[2].y)), (255,0,0), 1)
		# cv2.line(img, (int(square[2].x),int(square[2].y)), (int(square[3].x),int(square[3].y)), (255,0,0), 1)
		# cv2.line(img, (int(square[3].x),int(square[3].y)), (int(square[0].x),int(square[0].y)), (255,0,0), 1)

		# plt.imshow(img)
		# plt.title('Image')
		# plt.show()

	wordBBSquares = getSquarePoints(wordBB)
	for square in wordBBSquares[:1]:
		
		min_x = int(min([square[0].x, square[1].x, square[2].x, square[3].x]))
		max_x = int(max([square[0].x, square[1].x, square[2].x, square[3].x])) 
		
		min_y = int(min([square[0].y, square[1].y, square[2].y, square[3].y]))
		max_y = int(max([square[0].y, square[1].y, square[2].y, square[3].y]))
		
		crop_img = img[min_y:max_y, min_x:max_x]
		crop_img = cv2.resize(crop_img, (128, 128))
		crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		
		plt.title('cropped')
		plt.imshow(crop_gray, cmap='gray')
		plt.show()

		# cv2.line(img, (int(square[0].x),int(square[0].y)), (int(square[1].x),int(square[1].y)), (255,0,0), 1)
		# cv2.line(img, (int(square[1].x),int(square[1].y)), (int(square[2].x),int(square[2].y)), (0,255,0), 1)
		# cv2.line(img, (int(square[2].x),int(square[2].y)), (int(square[3].x),int(square[3].y)), (0,0,255), 1)
		# cv2.line(img, (int(square[3].x),int(square[3].y)), (int(square[0].x),int(square[0].y)), (255,255,0), 1)
		
		# plt.imshow(img)
		# plt.title('Image')
		# plt.show()

# for name in im_names[:1]:
	# showImageWordsAndChars(name)

dataset = CharsDataset()

for img,label in dataset:
	# here comes your training loop
	plt.title(img)
	plt.imshow(label, cmap='gray')
	plt.show()
	pass