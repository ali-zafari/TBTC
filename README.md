# Transformer-based Transform Coding (TBTC)
Pytorch implementation of four neural image compression models discussed in the paper. This unofficial implementation is relied on the [CompressAI](https://github.com/InterDigitalInc/CompressAI) pytorch library and is also wrapped by [Lightning](https://github.com/Lightning-AI/lightning).

4 models are implemented in [`compressai/models/qualcomm.py`](compressai/models/qualcomm.py)
- Conv-Hyperprior
- Conv-ChARM
- SwinT-Hyperprior
- SwintT-ChARM

## Usage
A local clone of the CompressAI is provied to make the model integration easier.

### Installation
In a virtual environment follow the steps below (verified on Ubuntu):
```bash
git clone https://github.com/ali-zafari/tbtc TBTC
cd TBTC
pip install -U pip
pip install -e .
pip install lightning==2.0.2
pip install tensorboard
```
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
The design paradigm of [CompressAI](https://github.com/InterDigitalInc/CompressAI) is closely followed which results to modifications/additions in the following directories:
```
compressai
    |__losses
    |   |----rate_distortion.py
    |
    |__layers
    |   |----swin.py
    |
    |__models
    |   |----qualcomm.py
    |
    |__zoo
        |----image.py
```

## Citations
#### Repositories
- [CompressAI](https://github.com/InterDigitalInc/CompressAI): Neural comporession library
- [SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM): Unofficial Tensorflow implementation
- [STF](https://github.com/Googolxx/STF): window-based neural image compression
- [Lightning](https://github.com/Lightning-AI/lightning): Pytorch framework for training abstraction

#### Publications
```
@article{begaint2020compressai,
	title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
	author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
	year={2020},
	journal={arXiv preprint arXiv:2011.03029},
}
```
```
@article{begaint2020compressai,
	title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
	author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
	year={2020},
	journal={arXiv preprint arXiv:2011.03029},
}
```