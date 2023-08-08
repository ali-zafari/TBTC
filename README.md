# Transformer-based Transform Coding (TBTC)
PyTorch implementation of four neural image compression models of [**Transformer-based Transform Coding**](https://openreview.net/forum?id=IDwN6xjHnK8) preseneted at *ICLR 2022*.

4 models are implemented in [`compressai/models/qualcomm.py`](compressai/models/qualcomm.py): *Conv-Hyperprior*, *Conv-ChARM*, *SwinT-Hyperprior*, *SwinT-ChARM*, as shown below.

|**Conv-Hyperprior**|**Conv-ChARM**|
|:---:|:---:|
|<img src="assets/convhyperprior.png" width="95%" alt="conv-hyperprior">|<img src="assets/convcharm.png" width="95%" alt="conv-charm">|
|**SwinT-Hyperprior**|**SwinT-ChARM**|
|<img src="assets/swinthyperprior.png" width="95%" alt="swint-hyperprior">|<img src="assets/swintcharm.png" width="95%" alt="swint-charm">|


Models' configurations are defined in a python dictionay object named [`cfgs`](compressai/zoo/image.py#L271) in [compressai/zoo/image.py](compressai/zoo/image.py) as described in [Section A.3 of Transformer-based Transform Coding](https://openreview.net/pdf?id=IDwN6xjHnK8).

## Pretrained Models
Models are trained with rate-distortion objective of $R+\lambda D$ with fixed $\lambda$ value mentioned in the following table.
| Model | Size | #Param | $\lambda$ | checkpoint | TensorBoard.dev<br>logs | Kodak <br> [bpp] / [dB]| GMACs [^1] <br> (ENC/DEC) | #steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-Hyperprior	 | "M" | 21.4M |0.01| [link](https://drive.google.com/file/d/1RyDmDDqrIwkVvVvH3HlPfwAoK4jmWY97/view?usp=sharing) | [link](https://tensorboard.dev/experiment/gUG6uE0QQdqc05EG8zCXNA/#scalars&runSelectionState=eyJjb252LWNoYXJtIjpmYWxzZSwiY29udi1oeXBlcnByaW9yIjp0cnVlLCJzd2ludC1jaGFybSI6ZmFsc2UsInN3aW50LWh5cGVycHJpb3IiOmZhbHNlfQ%3D%3D&tagFilter=valid) | 0.43 / 33.03| 99 / 350 | 2M |
| Conv-ChARM	 | "M" | 29.1M | 0.01 | [link](https://drive.google.com/file/d/1_UwLe_hwxKDnT-Nd4jrBTZdqFNygK0j2/view?usp=sharing) | [link](https://tensorboard.dev/experiment/gUG6uE0QQdqc05EG8zCXNA/#scalars&runSelectionState=eyJjb252LWNoYXJtIjp0cnVlLCJjb252LWh5cGVycHJpb3IiOmZhbHNlLCJzd2ludC1jaGFybSI6ZmFsc2UsInN3aW50LWh5cGVycHJpb3IiOmZhbHNlfQ%3D%3D&tagFilter=valid) | 0.41 / 33.17| 111 / 361 | 2M |
| SwinT-Hyperprior	 | "M" | 24.7M  | 0.01| [link](https://drive.google.com/file/d/1FS5t5kOZloUwdJr-DEOP-XSFYUdNj5gq/view?usp=sharing) | [link](https://tensorboard.dev/experiment/gUG6uE0QQdqc05EG8zCXNA/#scalars&runSelectionState=eyJjb252LWNoYXJtIjpmYWxzZSwiY29udi1oeXBlcnByaW9yIjpmYWxzZSwic3dpbnQtY2hhcm0iOmZhbHNlLCJzd2ludC1oeXBlcnByaW9yIjp0cnVlfQ%3D%3D&tagFilter=valid) | 0.38 / 32.67 |  99 / 99 |2M |
| SwinT-ChARM	 | "M" | 32.4M | 0.01 | [link](https://drive.google.com/file/d/1i7Q1S74b2f2kar76dVSPRH2C-Nu0p6ou/view?usp=sharing) | [link](https://tensorboard.dev/experiment/gUG6uE0QQdqc05EG8zCXNA/#scalars&runSelectionState=eyJjb252LWNoYXJtIjpmYWxzZSwiY29udi1oeXBlcnByaW9yIjpmYWxzZSwic3dpbnQtY2hhcm0iOnRydWUsInN3aW50LWh5cGVycHJpb3IiOmZhbHNlfQ%3D%3D&tagFilter=valid) | 0.37 / 33.07 | 110 / 110 | 2M |

[^1]: per input image size of 768x512

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



## Code Structure
This unofficial PyTorch implementation follows the [CompressAI](https://github.com/InterDigitalInc/CompressAI) code structure and then is wrapped by the [Lightning](https://github.com/Lightning-AI/lightning) framework. Tensorflow implementation of [SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM) is used as the reference.

The design paradigm of [CompressAI](https://github.com/InterDigitalInc/CompressAI) is closely followed which results to modifications/additions in the following directories. [Lightning](https://github.com/Lightning-AI/lightning)-based python files are also shown below:
```
|---compressai
|    |---losses
|    |    ├───rate_distortion.py       rate-disortion loss
|    |---layers
|    |    ├───swin.py                  blocks needed by TBTC models
|    |---models
|    |    ├───qualcomm.py              TBTC models
|    |---zoo
|         ├───image.py                 model creation based on config
|
├───lit_config.py                      configuration file
├───lit_data.py                        lighting data-module   
├───lit_model.py                       lightning module
├───lit_train.py                       main script to start/resume training
```

## References/Citations
#### Repositories
- [CompressAI](https://github.com/InterDigitalInc/CompressAI): Neural comporession library in PyTorch (by [InterDigital](https://www.interdigital.com/))
- [NeuralCompression](https://github.com/facebookresearch/NeuralCompression): Neural comporession library in PyTorch (by [Meta](https://opensource.fb.com/))
- [SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM): Unofficial Tensorflow implementation
- [STF](https://github.com/Googolxx/STF): Window-based attention in neural image compression
- [Lightning](https://github.com/Lightning-AI/lightning): PyTorch framework for training abstraction

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
