import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np

PATH = '/home/annie/Escritorio'
INPATH = PATH + '/salida'
OUTPATH = PATH + '/entrada'
CHPATH = PATH + '/checkpoints'

result = next(os.walk(INPATH))[2]

n = 500

train_n = round(n*0.80)

randurls = np.copy(result)

np.random.seed(23)
np.random.shuffle(randurls)

tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(result), len(tr_urls), len(ts_urls))

#BUFFER_SIZE = 400
#BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


#escalar imagenes
def resize(inimg, tgimg, height, width):
	inimg = tf.image.resize(inimg, [height, width])
	tgimg = tf.image.resize(tgimg, [height, width])
	return inimg, tgimg


#normalizar 
def normalize(inimg, tgimg, height, width):
	inimg = (inimg / 127.5) - 1
	tgimg = (tgimg / 127.5) - 1
	return inimg, tgimg

#AUMENTACION DE DATOS 
def random_jitter(inimg, tgimg):
	inimg, tgimg = resize(inimg, tgimg, 286,286)
	stacked_image = tf.stack([inimg, tgimg], axis=0)
	cropped_image = tf.image.random_crop(stacked_image, size = [2, IMG_HEIGHT, IMG_WIDTH, 3])
	inimg, tgimg = crooped_image[0], cropped_image[1]
	if tf.random.uniform(()) > 5:
		inimg = tf.image.flip_left_right(inimg)
		tgimg = tf.image.flip_left_right(tgimg)
	return inimg, tgimg


def load_image(filename, augment = True):
	inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32)[..., :3]
	tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH + '/' + filename)), tf.float32)[..., :3]
	inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)
	
	if augment:
		inimg, tgimg = random_jitter(inimg, tgimg)
	inimg, tgimg = normalize(inimg, tgimg)
	return inimg, tgimg

def load_train_image(filname):
	return load_image(filname, True)

def load_test_image(filname):
	return load_image(filname, False)

#plt.imshow(((load_train_image(randurls[0])[0]) + 1) / 2)

train_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
train_dataset = train_dataset.map(load_train_image)
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(result), len(tr_urls), len(ts_urls))

#BUFFER_SIZE = 400
#BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
#escalar imagenes
def resize(inimg, tgimg, height, width):
	inimg = tf.image.resize(inimg, [height, width])
	tgimg = tf.image.resize(tgimg, [height, width])
	return inimg, tgimg

#normalizar 
def normalize(inimg, tgimg, height, width):
	inimg = (inimg / 127.5) - 1
	tgimg = (tgimg / 127.5) - 1
	return inimg, tgimg


 

