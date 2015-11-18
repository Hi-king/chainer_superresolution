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
parser.add_argument("original_img")
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
print(PATCH_SHAPE)


#class WholeImageConverter(object):
#    def convert(x):


# test
original_img = cv2.imread(args.original_img)

mini_img = cv2.resize(
    original_img,
    (original_img.shape[1]/2, original_img.shape[0]/2)
)
noisy_img = cv2.resize(mini_img, (original_img.shape[1], original_img.shape[0]))

data = numpy.array([noisy_img.transpose((2,0,1))], dtype=numpy.float32)
if args.gpu >= 0:
    data = chainer.cuda.to_gpu(data)
predicted = model.predict(data)
bgr = numpy.array(chainer.cuda.to_cpu(predicted[0]), dtype=int)
cv2.imwrite("atonce.png", bgr.transpose((1, 2, 0)))
cv2.imwrite("noisy.png", noisy_img)

exit(0)




for x in xrange(100, 300):
    for y in xrange(100, 300):
        print(noisy_img.shape)
        input_noisy = noisy_img[
            y-(PATCH_SHAPE[1]-1)/2:y+(PATCH_SHAPE[1]+1)/2,
            x-(PATCH_SHAPE[0]-1)/2:x+(PATCH_SHAPE[0]+1)/2,
            :
        ]
        print(input_noisy.shape)
        data = numpy.array([input_noisy.transpose((2,0,1))], dtype=numpy.float32)
        if args.gpu >= 0:
            data = chainer.cuda.to_gpu(data)
        predicted = model.predict(data)
        bgr = numpy.array(chainer.cuda.to_cpu(predicted[0]), dtype=int)
        print(bgr)
        zero_img[y, x] = bgr
        raw_img[y, x] = noisy_img[y, x]
cv2.imwrite("converted.png", args.converted_img)

