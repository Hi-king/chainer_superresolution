CNN Superresolution
========================

How to train
--------------

2x upsampling

```
python train.py --gpu=1 simple3layer imgs/double/train imgs/double/evaluation
```

with Single (noisy, clean) pair

```
python train_single.py conv3layer imgs/miku_CC_BY-NC.jpg imgs/miku_CC_BY-NC.jpg
```

