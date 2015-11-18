CNN Superresolution
========================

How to train
--------------

2x upsampling

```
python train.py --gpu=1 simple3layer imgs/double/directory_contains_imgs
```


Evaluation

```
python evaluation.py --gpu=1 output/simple3layer_9900001.dump imgs/miku_CC_BY-NC.jpg
open noisy.png # downsampled img
open converted.png # converted from noisy.png
```
