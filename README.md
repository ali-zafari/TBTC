# Transformer-based Transform Coding (TBTC)
Pytorch implementation of four neural image compression models of [**Transformer-based Transform Coding**](https://openreview.net/forum?id=IDwN6xjHnK8) preseneted at *ICLR 2022*.

This unofficial Pytorch implementation follows the [CompressAI](https://github.com/InterDigitalInc/CompressAI) code structure and then is wrapped by the [Lightning](https://github.com/Lightning-AI/lightning) framework. Tensorflow implementation of [SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM) is used as the reference.


4 models are implemented in [`compressai/models/qualcomm.py`](compressai/models/qualcomm.py)
- **Conv-Hyperprior**
- **Conv-ChARM**
- **SwinT-Hyperprior**
- **SwinT-ChARM**

Models' configurations are defined in a python dictionay object named [`cfgs`](compressai/zoo/image.py#L271) in [compressai/zoo/image.py](compressai/zoo/image.py).

## Usage
A local clone of the CompressAI is provided to make the model integration easier.

### Installation
In a virtual environment follow the steps below (verified on Ubuntu):
```bash
git clone https://github.com/ali-zafari/TBTC TBTC
cd TBTC
pip install -U pip
pip install -e .
pip install lightning==2.0.2
pip install tensorboard
```
### Datasets
[CLIC-2020](https://www.tensorflow.org/datasets/catalog/clic) is used for training, described below.
- Training
  - `1631` images with resolution of at least 256x256 pixels chosen from union of `Mobile/train` and `Professional/train`
- Validation
  - `32` images with resolution of at least 1200x1200 pixels chosen from `Professional/valid`

[Kodak](https://r0k.us/graphics/kodak/) test set is used to evaluate the final trained model.
- Test
  - `24` RGB images of size 512x768 pixels

All three data subsets described above can be downloaded from this [link](https://drive.google.com/file/d/1g-qWy_i6kTVGDBYh1ol0corugyT-xVJG/view?usp=sharing) (5.8GB).

### Training
All the configurations regarding dataloader, training strategy, and etc should be set in the `lit_config.py` followed by the command:
```bash
python lit_train.py --comment "simple comment for the experiment"
```

### Evaluation
To evaluate a saved checkpoint of a model, `compressai.utils.eval` is used. An example to test the rate-distoriton perfomance of a SwinT-ChARM checkpoint:

```bash
python -m compressai.utils.eval_model checkpoint path/to/data/directory  -a zyc2022-swint-charm --cuda -v -p path/to/a/checkpoint
```

## Pretrained Models
Models are trained with rate-distortion objective of $R+\lambda D$ with fixed $\lambda$ value mentioned in the following table.
| Model | Size | #Params | $\lambda$ | Kodak <br> bitrate | Kodak <br> distortion| GMACs [^1] <br> (ENC/DEC) | model | #steps | logs |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| Conv-Hyperprior	 | "M" | 21.4M |0.01| 0.43 bpp | 32.93 dB | 99 / 350 | [link](https://drive.google.com/file/d/1whqdbRN7uVpacGrzO5SLv8F9rTurReQ2/view?usp=drive) | 1M | [link](https://tensorboard.dev/experiment/ecKVRhlRTg6hDAGRbeiZTw/#scalars) |
| Conv-ChARM	 | "M" | 29.1M | 0.01 | 0.42 bpp | 33.10 dB | 111 / 361 | [link](https://drive.google.com/file/d/1OJ5nJFSdZNtAiqnBCK272DI8YHmOPwGa/view?usp=drive_link) | 1M | [link](https://tensorboard.dev/experiment/dvb7lh6rT7me1UMJOw8kaw/#scalars) |
| SwinT-Hyperprior	 | "M" | 24.7M | 0.01 | 0.37 bpp | 14.56 dB |  99 / 99 | [link](https://drive.google.com/file/d/16zJX3zj-742UPI4zWwvaTgSibuHWuDj2/view?usp=sharing) | 1M | [link](https://tensorboard.dev/experiment/wk66DS80QJKZlSQvTMEPNg/#scalars) |
| SwinT-ChARM	 | "M" | 32.4M | 0.01 | --- | --- | 110 / 110 | --- | --- | --- |

[^1]: per input image size of 768x512

## Code Structure
The design paradigm of [CompressAI](https://github.com/InterDigitalInc/CompressAI) is closely followed which results to modifications/additions in the following directories. [Lightning](https://github.com/Lightning-AI/lightning)-based python files are also shown below:
```
|___compressai
|    |___losses
|    |    |---rate_distortion.py       rate-disortion loss
|    |___layers
|    |    |---swin.py                  blocks needed by TBTC models
|    |___models
|    |    |---qualcomm.py              TBTC models
|    |___zoo
|         |---image.py                 model creation based on config
|
|---lit_config.py                      configuration file
|---lit_data.py                        lighting data-module   
|---lit_model.py                       lightning module
|---lit_train.py                       main script to start/resume training
```

## References/Citations
#### Repositories
- [CompressAI](https://github.com/InterDigitalInc/CompressAI): Neural comporession library in Pytorch (by InterDigital)
- [NeuralCompression](https://github.com/facebookresearch/NeuralCompression): Neural comporession library in Pytorch (by Meta)
- [SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM): Unofficial Tensorflow implementation
- [STF](https://github.com/Googolxx/STF): Window-based attention in neural image compression
- [Lightning](https://github.com/Lightning-AI/lightning): Pytorch framework for training abstraction

#### Publications
```
@inproceedings{zhu2022tbtc,
  title={Transformer-based transform coding},
  author={Zhu, Yinhao and Yang, Yang and Cohen, Taco},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@article{begaint2020compressai,
	title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
	author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
	year={2020},
	journal={arXiv preprint arXiv:2011.03029},
}
```
