# NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views

[[Paper]](https://arxiv.org/abs/2211.16431) [[Website]](https://vita-group.github.io/NeuralLift-360/)

## Pipeline

![](./docs/static/media/framework-crop-1.b843bf7d1c3c29c01fb2.jpg)

## Environment

`pip install -r requirements.txt` will do the job.

## Data Preparation

In our experiments, we use the depth from [Boost Your Own depth](https://github.com/compphoto/BoostingMonocularDepth) together with [LeRes](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS).

The colab notebook to export depth in numpy can be found [here](https://colab.research.google.com/drive/15YCsqaO6l94HueVwPQgHqVVDUJzdOEO5?usp=sharing).

## Training

## Testing

## Acknowledgement

Codebase based on https://github.com/ashawkey/stable-dreamfusion . Thanks [Jiaxiang Tang](https://me.kiui.moe/) for sharing and the insightful discussions!

## Citation

If you find this repo is helpful, please cite:

```

@InProceedings{Xu_2022_neuralLift,
author = {Xu, Dejia and Jiang, Yifan and Wang, Peihao and Fan, Zhiwen and Wang, Yi and Wang, Zhangyang},
title = {NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views},
journal={arXiv preprint arXiv:2211.16431},
year={2022}
}

```


