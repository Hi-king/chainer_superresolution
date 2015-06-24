#!/usr/bin/env python
import chainer
import argparse
import cv2
import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("modelpath")
parser.add_argument("noisy_img")
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.init(args.gpu)
    print args.gpu
    models.simple3layer.chainer.cuda.init(args.gpu)
    models.conv3layer.chainer.cuda.init(args.gpu)
    models.conv3layer_large.chainer.cuda.init(args.gpu)

with open(args.modelpath) as f:
    model = pickle.load(f)
PATCH_SHAPE = model.PATCH_SHAPE


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
        bgr = numpy.array(chainer.cuda.to_cpu(predicted[0]), dtype=int)
        print(bgr)
        zero_img[y, x] = bgr
        raw_img[y, x] = noisy_img[y, x]
cv2.imwrite("converted.png", zero_img)
cv2.imwrite("converted_raw.png", raw_img)
