from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# TODO:
# * mAP50, mAP50-95, precision, recall. NOTE: torchmetrics

class TRetina(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = retinanet_resnet50_fpn_v2(num_classes=num_classes)
        self.optimizer = None

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        print('Saving Model...')
        torch.save(self.model.state_dict(), path)
        print(f'Model saved at {path}')

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def info(self, verbose=False):
        # TODO: Return a more verbose model info
        print(self.model.__str__())


class PLRetina(pl.LightningModule):
    def __init__(self, model: TRetina):
        super().__init__()
        self.model = model
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.valid_map = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, targets = batch
        image, targets = list(image), list(targets)  # TODO: collate_fn
        loss_dict = self.model(image, targets)
        loss_sum = sum(loss for loss in loss_dict.values())
        print(loss_dict.values())
        loss_dict['loss'] = loss_sum
        self.log('train_loss', loss_sum, logger=False)
        self.training_step_outputs.append(loss_dict)
        return loss_dict

    def on_train_epoch_end(self):
        # Getting outputs
        outputs = self.training_step_outputs

        # Logging losses
        writer = self.logger.experiment

        # Generic loss plotter
        losses = dict()
        for output in outputs:
            for key, value in output.items():
                if losses.get(key):
                    losses[key].append(value)
                else:
                    losses[key] = [value]


        nlosses = {key: torch.stack(value).mean() for key, value in losses.items()}

        for key, value in nlosses.items():
            writer.add_scalar(
                f"Training/{key}",
                value,
                self.current_epoch)

        # free up the memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image, targets = batch
        image, targets = list(image), list(targets)  # TODO: collate_fn
        predections = self.model(image, targets)
        return {'preds': predections, 'target': targets}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        self.valid_map.update(outputs['preds'], outputs['target'])

    def on_validation_epoch_end(self):
        # Logging losses
        writer = self.logger.experiment

        epoch_map = self.valid_map.compute()

        map_m = epoch_map['map']
        map_50 = epoch_map['map_50']
        mar_100 = epoch_map['mar_100']
        # rec = epoch_map['']
        # map_75 = epoch_map['map_75']
        # map_90 = epoch_map['map_90']

        self.log('valid_map', map_m)

        writer.add_scalar(
            f"Validation/Mean average precision",
            map_m,
            self.current_epoch)
        writer.add_scalar(
            f"Validation/Map@50",
            map_50,
            self.current_epoch)
        writer.add_scalar(
            f"Validation/Mar@100",
            mar_100,
            self.current_epoch)

        self.valid_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def info(self, verbose=False):
        max_depth = -1 if verbose else 1
        return ModelSummary(self, max_depth=max_depth)
