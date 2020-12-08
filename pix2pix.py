import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os 

PATH = '/home/annie/Escritorio/'
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


 

