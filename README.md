# NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views

[[Paper]](https://arxiv.org/abs/2211.16431) [[Website]](https://vita-group.github.io/NeuralLift-360/)

## News

- **23.03.12 Basic workflow is released! [Gradio App](#gradio) released!** 

We're working on rolling out new exciting features soon and will prepare tutorials and better interface using gradio. Stay tuned!

## Pipeline

![](./docs/static/media/framework-crop-1.b843bf7d1c3c29c01fb2.jpg)

## Environment

`pip install -r requirements.txt` will do the job.

`pip install gradio` if you want to run the [Gradio App](#gradio).

## Data Preparation

In our experiments, we use the depth from [Boost Your Own depth](https://github.com/compphoto/BoostingMonocularDepth) together with [LeRes](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS).

The colab notebook to export depth in numpy can be found [here](https://colab.research.google.com/drive/15YCsqaO6l94HueVwPQgHqVVDUJzdOEO5?usp=sharing).

The foreground mask can be obtained using this repo: https://github.com/Ir1d/image-background-remove-tool


## Gradio

```bash
python gradio_app.py
# add `--share` if you want an external public link
```

We prepare a Gradio App as well!

Note: Currently, we load config from a pre-defined yaml file and we'll update this later. Also, it's a bit slower compared to directly running the training script because we have to render during training.

## Training

We prepare the config files in the yaml file in `configs` directory.

```bash
python main.py --config configs/cabin.yaml
```

## Testing

After training process finishes, the code will automatically generates a video named `lift_ep0100_rgb.mp4`

You can filter out all these videos in the log folder and sort them by time using the following code.

```bash
find ./ -name lift_ep0100_rgb.mp4 -printf "%T@ %Tc %p\n" | sort -n  
```

If you want to run test only, change `test: False` to `test: True` in the yaml config file.


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


