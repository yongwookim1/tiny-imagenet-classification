# Pytorch-Tiny-ImageNet

### Installation

```
pip install -r requirements.txt
```

### Dataset

`python prepare_dataset.py` will download and preprocess tiny-imagenet dataset.
In the original dataset, there are 200 classes, and each class has 500 images.
However, in test dataset there are no labels, so I split the validation dataset into validation and test dataset. (25 per class)
Probably not the best train(500), val(25), test(25) splitting method, but I think it's good enough for this project to evaluate transfer learning.

The dataset is then resized from 64x64 to 224x224.

You can also download dataset here.
- [tiny-imagenet-200](https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip)
- [tiny-imagenet-200 resized to 224x224](https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/tiny-224.zip)

### Summary

Goal of this project is to evaluate transfer learning on tiny-imagenet dataset.

Tiny-ImageNet dataset has images of size 64x64, but ImageNet dataset is trained on 224x224 images.
To match the input size, I resized tiny-imagenet dataset to 224x224 and trained on pretrained weight from ImageNet.

Finetune few layers, and use pretrained weight from 224x224 trained model to retrain 64x64 image on ResNet18

### Test Result

| Model    | Test Result | Input size | pretrained weight |
| -------- | ----------- | ---------- | ----------------- |
| AlexNet  | 35.88%      | 64x64      | ImageNet          |
| ResNet18 | 53.58%      | 64x64      | ImageNet          |
| ResNet18 | 69.62%      | 224x224    | ImageNet          |

### Acknowledgement

This repository is modified and adapted on this repository.
[pytroch-tiny-imagenet](https://github.com/tjmoon0104/pytorch-tiny-imagenet)
