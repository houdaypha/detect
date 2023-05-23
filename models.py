import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import pytorch_lightning as pl
from dataset import CustomData

# FasterRCNN Model

class TFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        print('Saving Model...')
        torch.save(self.model.state_dict(), path)
        print(f'Model saved at {path}')

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class PLFasterRCNN(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer


class Model:
    def __init__(self, model: str, num_classes, device='cpu'):
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
                train_dataloaders=data.train_dataloader,
                val_dataloaders=data.val_dataloader)

        elif self.type == 'torch':
            raise NotImplementedError
        elif self.type == 'yolo':
            raise NotImplementedError
        
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
