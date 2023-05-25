from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from dataset import CustomData
from collections import OrderedDict
import utils

# FasterRCNN Model


class TFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)

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


class PLFasterRCNN(pl.LightningModule):
    def __init__(self, model: FasterRCNN):
        super().__init__()
        self.model = model
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        image, targets = batch
        image, targets = list(image), list(targets)  # TODO: collate_fn
        # 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
        loss_dict = self.model(image, targets)
        loss_sum = sum(loss for loss in loss_dict.values())
        loss_dict['loss'] = loss_sum
        self.log('train_loss', loss_sum, logger=False)
        self.training_step_outputs.append(loss_dict)
        return loss_dict

    def on_train_epoch_end(self):
        # Getting outputs
        outputs = self.training_step_outputs

        # Logging losses
        writer = self.logger.experiment
        tloss = torch.stack([x["loss"] for x in outputs]).mean()
        closs = torch.stack([x["loss_classifier"] for x in outputs]).mean()
        rloss = torch.stack([x["loss_box_reg"] for x in outputs]).mean()
        rpnloss = torch.stack([x["loss_rpn_box_reg"] for x in outputs]).mean()
        oloss = torch.stack([x["loss_objectness"] for x in outputs]).mean()
        writer.add_scalar(
            'Training/Total Loss',
            tloss,
            self.current_epoch)
        writer.add_scalar(
            'Training/Classifier Loss',
            closs,
            self.current_epoch)
        writer.add_scalar(
            'Training/Regressor loss',
            rloss,
            self.current_epoch)
        writer.add_scalar(
            'Training/Region Proposal Network loss',
            rpnloss,
            self.current_epoch)
        writer.add_scalar(
            'Training/Objectness loss',
            oloss,
            self.current_epoch)

        # free up the memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image, targets = batch
        image, targets = list(image), list(targets)
        loss_dict, _ = utils.eval_forward(self.model, image, targets)
        loss_sum = sum(loss for loss in loss_dict.values())

        # Validation loss dictionary
        val_loss_dict = dict()
        for k, v in loss_dict.items():
            val_loss_dict[f'val_{k}'] = v
        val_loss_dict['loss'] = loss_sum

        self.log('val_loss', loss_sum, logger=False)
        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict

    def on_validation_epoch_end(self):
        # Getting outputs
        outputs = self.validation_step_outputs

        # Logging losses
        writer = self.logger.experiment
        tloss = torch.stack([x["loss"] for x in outputs]).mean()
        closs = torch.stack([x["val_loss_classifier"] for x in outputs]).mean()
        rloss = torch.stack([x["val_loss_box_reg"] for x in outputs]).mean()
        rpnloss = torch.stack(
            [x["val_loss_rpn_box_reg"] for x in outputs]).mean()
        oloss = torch.stack([x["val_loss_objectness"] for x in outputs]).mean()
        writer.add_scalar(
            'Validation/Total Loss',
            tloss,
            self.current_epoch)
        writer.add_scalar(
            'Validation/Classifier Loss',
            closs,
            self.current_epoch)
        writer.add_scalar(
            'Validation/Regressor loss',
            rloss,
            self.current_epoch)
        writer.add_scalar(
            'Validation/Region Proposal Network loss',
            rpnloss,
            self.current_epoch)
        writer.add_scalar(
            'Validation/Objectness loss',
            oloss,
            self.current_epoch)

        # free up the memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def info(self, verbose=False):
        max_depth = -1 if verbose else 1
        return ModelSummary(self, max_depth=max_depth)


class Model:
    r"""

    Args
        model (str): type of model
        num_classes (int): including the background
        device (str | list): model device
    """

    def __init__(self, model: str, num_classes: int, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = model

        if self.model == 'fasterrcnn':
            if self.device == 'gpu' or isinstance(device, list):
                self.type = 'pl'
                tmodel = TFasterRCNN(self.num_classes)
                self.model = PLFasterRCNN(tmodel.model)
            elif self.device == 'cpu':
                self.type = 'torch'
                self.model = TFasterRCNN(self.num_classes)

        elif self.model == 'ssd':
            if self.device == 'gpu' or isinstance(device, list):
                self.type = 'pl'
                raise NotImplementedError
            elif self.device == 'cpu':
                self.type = 'torch'
                raise NotImplementedError

        elif self.model == 'yolo':
            self.type = 'ultralytics'
            raise NotImplementedError

        else:
            raise Exception(f'Model {self.model} not supported')

    def train(self, data: str, epochs=10, batch=4, shuffle=True, workers=4):
        if self.type == 'pl':
            data: CustomData = CustomData(data, batch, shuffle, workers)
            # Print dataset information
            print(data)

            # Define trainer
            if self.device == 'gpu':
                trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu')
            elif isinstance(self.device, list):
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    strategy="ddp",
                    accelerator="gpu",
                    devices=self.device)

            # Train pl model
            trainer.fit(
                model=self.model,
                train_dataloaders=data.train_loader,
                val_dataloaders=data.valid_loader)

        elif self.type == 'torch':
            raise NotImplementedError
        elif self.type == 'yolo':
            raise NotImplementedError

    def export(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def predict(self, source):
        """
        Perform prediction using the model

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
        """
        raise NotImplementedError

    def info(self, verbose=True):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)


def main():
    model = Model('fasterrcnn', 2, 'gpu')
    model.train('./conf/torch.yaml', epochs=10, batch=2, workers=4)


if __name__ == '__main__':
    main()
