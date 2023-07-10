import math
from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer

from lightning.pytorch import LightningModule
from lit_config import Config


class CompressorLit(LightningModule):
    """Lightning-based compression model
    """

    def __init__(self, net, net_lr, aux_lr, lmbda, beta=None, gradient_clip_norm=1.0, comment=None):
        super().__init__()
        self.net = net
        self.net_lr = net_lr
        self.aux_lr = aux_lr
        self.gradient_clip_norm = gradient_clip_norm
        self.comment = comment

        self.criterion = RateDistortionLoss(lmbda=lmbda, beta=beta)
        self.automatic_optimization = False

        self.save_configuration()

    def save_configuration(self):
        self.hparams["a-Model"] = dict()
        self.hparams["b-Trainer"] = dict()
        self.hparams["c-Data"] = dict()

        self.hparams["a-Model"]["a-comment"] = self.comment
        for k, v in dict(vars(Config.Model)).items():
            if not k.startswith("_"):
                self.hparams["a-Model"][k] = v
        for k, v in dict(vars(Config.Trainer)).items():
            if not k.startswith("_"):
                self.hparams["b-Trainer"][k] = v
        for k, v in dict(vars(Config.Data)).items():
            if not k.startswith("_"):
                self.hparams["c-Data"][k] = v

        self.save_hyperparameters(ignore=['net', 'net_lr', 'aux_lr', 'lmbda', 'beta', 'gradient_clip_norm', 'comment'])

    def on_train_start(self):
        with open(f'{self.trainer.log_dir}/model.txt', mode='w') as f:
            f.write(str(self.net))

    def configure_optimizers(self):
        conf = {
            "net": {"type": "Adam", "lr": self.net_lr},
            "aux": {"type": "Adam", "lr": self.aux_lr},
        }

        optimizers = net_aux_optimizer(self.net, conf)

        scheduler_step_size = int(0.9 * self.trainer.estimated_stepping_batches)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizers["net"], step_size=scheduler_step_size, gamma=0.1)

        return ({"optimizer": optimizers["net"], "lr_scheduler": lr_scheduler},
                {"optimizer": optimizers["aux"]},)

    def training_step(self, batch, batch_idx):
        net_opt, aux_opt = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        net_opt.zero_grad()
        aux_opt.zero_grad()

        out_net = self.net(batch)

        out_criterion = self.criterion(out_net, batch)
        self.manual_backward(out_criterion["loss"])
        if self.gradient_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip_norm)
        net_opt.step()

        aux_loss = self.net.aux_loss()
        self.manual_backward(aux_loss)
        aux_opt.step()

        lr_scheduler.step()

        log_info = {
            "train/loss": out_criterion["loss"].item(),
            "train/mse": out_criterion["mse_loss"].item() * 255 ** 2 / 3,
            "train/bpp": out_criterion["bpp_loss"].item(),
            "train/aux": aux_loss.item(),
            "train/psnr": -10 * math.log10(out_criterion["mse_loss"].item())
        }
        self.log_dict(log_info, sync_dist=True if len(Config.Trainer.devices) > 1 else False)

    def validation_step(self, batch, batch_idx):
        out_net = self.net(batch)
        out_criterion = self.criterion(out_net, batch)

        log_info = {
            "valid/loss": out_criterion["loss"],
            "valid/mse": out_criterion["mse_loss"] * 255 ** 2 / 3,
            "valid/bpp": out_criterion["bpp_loss"],
            "valid/aux": self.net.aux_loss(),
            "valid/psnr": -10 * math.log10(out_criterion["mse_loss"])
        }
        self.log_dict(log_info, sync_dist=True if len(Config.Trainer.devices) > 1 else False)

    def on_save_checkpoint(self, checkpoint):
        updated_state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            updated_state_dict[key[4:]] = checkpoint['state_dict'][key]  # remove "net." for compressai compatibility
        checkpoint['state_dict'] = updated_state_dict

    def on_load_checkpoint(self, checkpoint):
        updated_state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            updated_state_dict[f'net.{key}'] = checkpoint['state_dict'][key]  # add "net." for lightning compatibility
        checkpoint['state_dict'] = updated_state_dict
