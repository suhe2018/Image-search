#-*-coding:utf8-*-

#创建索引 字典的值将是计算图像的直方图
#使用此示例的字典最有意义，特别是出于解释目的。给定一个键，字典指向其他一些对象。当我们使用图像文件名作为关键字并使用直方图作为值时，
# 我们暗示给定的直方图  H用于量化并表示具有文件名K的图像
#（同样，您可以根据需要使此过程变得简单或复杂。更复杂的图像描述符利用术语频率 - 逆文档频率加权（tf-idf）和倒排索引）
from pyimagesearch.rgbhistogram import RGBHistogram
from imutils.paths import list_images
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required = True,
# 	help = "Path to the directory that contains the images to be indexed")
# ap.add_argument("-i", "--index", required = True,
# 	help = "Path to where the computed index will be stored")
# args = vars(ap.parse_args())

# 初始化索引字典以存储我们的量化的图像，字典的“键”是图像文件名和我们的计算功能的“值”
index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])

# use list_images to grab the image paths and loop over them
for imagePath in list_images("images"):
	# extract our unique image ID (i.e. the filename)
	k = imagePath[imagePath.rfind("/") + 1:]

	# load the image, describe it using our RGB histogram
	# descriptor, and update the index
	image = cv2.imread(imagePath)
	features = desc.describe(image)
	index[k] = features

# we are now done indexing our image -- now we can write our
# index to disk
f = open("index.cpickle2", "wb")
f.write(pickle.dumps(index))
f.close()

# show how many images we indexed
print("[INFO] done...indexed {} images".format(len(index)))