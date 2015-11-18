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
parser.add_argument("noisy_img")
parser.add_argument("original_img")
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

PATCH_SHAPE = (9, 9, 3)

if args.model == "simple3layer":
    model = models.simple3layer.Model(PATCH_SHAPE)
elif args.model == "conv3layer":
    model = models.conv3layer.Model(PATCH_SHAPE)
elif args.model == "conv3layer_large":
    PATCH_SHAPE = models.conv3layer_large.Model.PATCH_SHAPE
    model = models.conv3layer_large.Model()
else:
    exit(1)

if args.gpu >= 0:
    chainer.cuda.init(args.gpu)
    model.to_gpu()


def read_image_mean(noisy_path, num=10000):
    noisy_img = cv2.imread(noisy_path)
    noisy_img_big_width = numpy.concatenate(
        (numpy.fliplr(noisy_img), noisy_img, numpy.fliplr(noisy_img)),
        axis=1)
    noisy_img_big = numpy.concatenate(
        (numpy.flipud(noisy_img_big_width), noisy_img_big_width, numpy.flipud(noisy_img_big_width)),
        axis=0)

    original_size = noisy_img.shape
    summed_img = numpy.zeros(PATCH_SHAPE, dtype=float)
    for i in xrange(num):
        target_x = random.randint(0, original_size[0]-1)
        target_y = random.randint(0, original_size[1]-1)
        input_noisy = noisy_img_big[
            original_size[1]+target_y-(PATCH_SHAPE[1]-1)/2:original_size[1]+target_y+(PATCH_SHAPE[1]+1)/2,
            original_size[0]+target_x-(PATCH_SHAPE[0]-1)/2:original_size[0]+target_x+(PATCH_SHAPE[0]+1)/2
        ]
        summed_img += input_noisy
    return summed_img/i
    

def read_image_label(noisy_path, original_path, mean_img, num=10000):
    noisy_img = cv2.imread(noisy_path)
    noisy_img_big_width = numpy.concatenate(
        (numpy.fliplr(noisy_img), noisy_img, numpy.fliplr(noisy_img)),
        axis=1)
    noisy_img_big = numpy.concatenate(
        (numpy.flipud(noisy_img_big_width), noisy_img_big_width, numpy.flipud(noisy_img_big_width)),
        axis=0)

    original_img = cv2.imread(original_path)
    original_size = original_img.shape
    for i in xrange(num):
        target_x = random.randint(0, original_size[0]-1)
        target_y = random.randint(0, original_size[1]-1)
        target_bgr = original_img[target_y, target_x]
        #target_bgr = noisy_img[target_y, target_x]
        input_noisy = noisy_img_big[
            original_size[1]+target_y-(PATCH_SHAPE[1]-1)/2:original_size[1]+target_y+(PATCH_SHAPE[1]+1)/2,
            original_size[0]+target_x-(PATCH_SHAPE[0]-1)/2:original_size[0]+target_x+(PATCH_SHAPE[0]+1)/2
        ]
        yield input_noisy, target_bgr

# train
mean_img = read_image_mean(args.noisy_img)
data = []
label = []
for i, (img, bgr) in enumerate(read_image_label(args.noisy_img, args.original_img, mean_img, 10000000)):
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
