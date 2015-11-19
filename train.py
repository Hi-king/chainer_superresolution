#!/usr/bin/env python
"""
superresolution

"""
import cv2
import random
import argparse
import numpy
import chainer
import chainer.optimizers
import chainer.cuda
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("trains")
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

PATCH_SHAPE = (9, 9, 3)

if args.model == "simple3layer":
    model = models.simple3layer.Model(PATCH_SHAPE)
elif args.model == "conv3layer":
    model = models.conv3layer.Model(PATCH_SHAPE)
elif args.model == "fullconv3layer":
    model = models.fullconv3layer.Model(PATCH_SHAPE)
elif args.model == "fullconv5layer":
    model = models.fullconv5layer.Model()
    PATCH_SHAPE = model.PATCH_SHAPE
elif args.model == "conv3layer_large":
    PATCH_SHAPE = models.conv3layer_large.Model.PATCH_SHAPE
    model = models.conv3layer_large.Model()
else:
    print("no such model {}".format(args.model))
    exit(1)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    #chainer.cuda.init(args.gpu)
    model.to_gpu()


def read_image_label(trains_dir, num=10000):
    trains = []
    for filename in os.listdir(trains_dir):
        path = trains_dir +"/"+filename
        original_img = cv2.imread(path)
        if original_img.shape[0] % 2 == 1 or original_img.shape[1] % 2 == 1: continue
        
        mini_img = cv2.resize(original_img, (original_img.shape[1]/2, original_img.shape[0]/2))
        noisy_img = cv2.resize(mini_img, (original_img.shape[1], original_img.shape[0]))
        noisy_img_big_width = numpy.concatenate(
            (numpy.fliplr(noisy_img), noisy_img, numpy.fliplr(noisy_img)),
            axis=1)
        noisy_img_big = numpy.concatenate(
            (numpy.flipud(noisy_img_big_width), noisy_img_big_width, numpy.flipud(noisy_img_big_width)),
            axis=0)
        trains.append({
            "original": original_img,
            "noisy": noisy_img_big
        })

    for i in xrange(num):
        target_i = random.randint(0, len(trains)-1)
        original_img = trains[target_i]["original"]
        noisy_img_big = trains[target_i]["noisy"]
        original_size = original_img.shape
        target_y = random.randint(0, original_size[0]-1)
        target_x = random.randint(0, original_size[1]-1)
        target_bgr = original_img[target_y, target_x]
        input_noisy = noisy_img_big[
            original_size[0]+target_y-(PATCH_SHAPE[0]-1)/2:original_size[0]+target_y+(PATCH_SHAPE[0]+1)/2,
            original_size[1]+target_x-(PATCH_SHAPE[1]-1)/2:original_size[1]+target_x+(PATCH_SHAPE[1]+1)/2
        ] 
        #target_bgr = input_noisy[4, 4]
        yield input_noisy, target_bgr

# train
data = []
label = []
for i, (img, bgr) in enumerate(read_image_label(args.trains, 10000000)):
    #model.train(img.transpose((2,0,1)), bgr.reshape((3,1)))
    if i % 1000 == 1:
        data = numpy.array(data)
        label = numpy.array(label)
        if args.gpu >= 0:
            data = chainer.cuda.to_gpu(numpy.array(data, dtype=numpy.float32))
            label = chainer.cuda.to_gpu(numpy.array(label, dtype=numpy.float32))

        error = model.train(data, label)
        print("error\t{}\t{}".format(i, error))
        data = []
        label = []
    if i % 100000 == 1:
        with open("output/{}_{}.dump".format(args.model, i), "w+") as f:
            pickle.dump(model, f)
    else:
        data.append(img.transpose((2,0,1)))
        label.append(bgr.reshape(3,1))
