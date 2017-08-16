import caffe
import numpy as np 
import pandas as pd

caffe.set_mode_cpu()

net1 = caffe.Net("/home/ubuntu/caffe/workspace/Teacher-Student-Training/imagenet/alexnet-VGG/vgg16_train_val.prototxt", 
	"/home/ubuntu/caffe/workspace/preloaded_VGG.caffemodel", 
	caffe.TRAIN)

net2 = caffe.Net("/home/ubuntu/caffe/workspace/Teacher-Student-Training/imagenet/alexnet-VGG/vgg16_original.prototxt",
	"/home/ubuntu/caffe/workspace/VGG_ILSVRC_16_layers.caffemodel",
	caffe.TRAIN)

net1.params["conv_st_1_1"] == net2.params["conv1_1"]
net1.params["conv_st_2_1"] == net2.params["conv2_1"]

