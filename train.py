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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print sys.path
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
    models.simple3layer.PATCH_SHAPE = PATCH_SHAPE
    model = models.simple3layer.Model()
elif args.model == "conv3layer":
    models.conv3layer.PATCH_SHAPE = PATCH_SHAPE
    model = models.conv3layer.Model()
elif args.model == "conv3layer_large":
    PATCH_SHAPE = (13, 13, 3)
    models.conv3layer_large.PATCH_SHAPE = PATCH_SHAPE
    model = models.conv3layer_large.Model()
else:
    exit(1)

if args.gpu >= 0:
    chainer.cuda.init(args.gpu)
    model.to_gpu()

optimizer = chainer.optimizers.Adam()
#optimizer = chainer.optimizers.SGD(lr=0.00001)
#optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.9)
#optimizer.setup(model.collect_parameters())
optimizer.setup(model.collect_parameters())




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
    print noisy_img_big.shape

    original_img = cv2.imread(original_path)
    original_size = original_img.shape
    for i in xrange(num):
        target_x = random.randint(0, original_size[0]-1)
        target_y = random.randint(0, original_size[1]-1)
        #target_bgr = original_img[target_y, target_x]
        target_bgr = noisy_img[target_y, target_x]
        input_noisy = noisy_img_big[
            original_size[1]+target_y-(PATCH_SHAPE[1]-1)/2:original_size[1]+target_y+(PATCH_SHAPE[1]+1)/2,
            original_size[0]+target_x-(PATCH_SHAPE[0]-1)/2:original_size[0]+target_x+(PATCH_SHAPE[0]+1)/2
        ]
        #print input_noisy
        #print target_bgr
        yield input_noisy, target_bgr

# train
mean_img = read_image_mean(args.noisy_img)
data = []
label = []
for i, (img, bgr) in enumerate(read_image_label(args.noisy_img, args.original_img, mean_img, 1000000000)):
    #print img.shape
    #print bgr.shape

    #model.train(img.transpose((2,0,1)), bgr.reshape((3,1)))
    if i % 1000 == 1:
        print len(data)
        data = numpy.array(data)
        label = numpy.array(label)
        if args.gpu >= 0:
            data = chainer.cuda.to_gpu(numpy.array(data, dtype=numpy.float32))
            label = chainer.cuda.to_gpu(numpy.array(label, dtype=numpy.float32))

        model.train(data, label, optimizer)
        data = []
        label = []
    else:
        data.append(img.transpose((2,0,1)))
        #print "in"
        #print img[(PATCH_SHAPE[0]-1)/2, (PATCH_SHAPE[1]-1)/2, :]
        #print "out"
        #print bgr
        #label.append(img[(PATCH_SHAPE[0]-1)/2, (PATCH_SHAPE[1]-1)/2, :])
        label.append(bgr.reshape(3,1))

# test
noisy_img = cv2.imread(args.noisy_img)
zero_img = numpy.zeros(noisy_img.shape)
raw_img = numpy.zeros(noisy_img.shape)
for x in xrange(100, 300):
    for y in xrange(100, 300):
        input_noisy = noisy_img[
            y-(PATCH_SHAPE[1]-1)/2:y+(PATCH_SHAPE[1]+1)/2,
            x-(PATCH_SHAPE[0]-1)/2:x+(PATCH_SHAPE[0]+1)/2
        ]
        data = numpy.array([input_noisy.transpose((2,0,1))])
        if args.gpu >= 0:
            data = chainer.cuda.to_gpu(numpy.array(data, dtype=numpy.float32))
        predicted = model.predict(data)
        bgr = numpy.array(chainer.cuda.to_cpu(predicted.data[0]), dtype=int)
        print(bgr)
        zero_img[y, x] = bgr
        raw_img[y, x] = noisy_img[y, x]
cv2.imwrite("{}_out.png".format(args.model), zero_img)
cv2.imwrite("{}_raw.png".format(args.model), raw_img)
