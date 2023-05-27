import torch
import pytorch_lightning as pl
from ultralytics import YOLO
from dataset import CustomData
from fasterrcnn import TFasterRCNN, PLFasterRCNN
from ssd import TSSD, PLSSD
import utils

class Model:
    r"""

    Args
        model (str): type of model
        num_classes (int): including the background
        device (str | list): model device
    """

    def __init__(self, model: str, num_classes=2, device: str|int='cpu'):
        self._device = device
        self.num_classes = num_classes
        self._name = model
        self.model = None
        self._type = None

        if model == 'fasterrcnn':
            if self._device == 0 or isinstance(device, list):
                self._type = 'pl'
                tmodel = TFasterRCNN(self.num_classes)
                self.model = PLFasterRCNN(tmodel.model)
                self.model.to(self._device)
            elif self._device == 'cpu':
                self._type = 'torch'
                self.model = TFasterRCNN(self.num_classes)
                self.model.to(self._device)

        elif model == 'ssd':
            if self._device == 0 or isinstance(device, list):
                self._type = 'pl'
                print("Setting ssd on pytorch lightning")
                tmodel = TSSD(self.num_classes)
                self.model = PLSSD(tmodel.model)
                self.model.to(self._device)
            elif self._device == 'cpu':
                self._type = 'torch'
                self.model = TSSD(self.num_classes)
                self.model.to(self._device)

        elif model == 'yolo':
            self._type = 'yolo'
            self.model = YOLO('yolov8x.yaml')
            self.model.to(self._device)

        else:
            raise Exception(f'Model {self.model} not supported')

    def train(self, data: str, epochs=10, batch=4, shuffle=True, workers=4):
        if self._type == 'pl':
            data: CustomData = CustomData(data, batch, shuffle, workers)
            # Print dataset information
            print(data)

            # Define trainer
            if self._device == 0:
                trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu')
            elif isinstance(self._device, list):
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    strategy="ddp",
                    accelerator="gpu",
                    devices=self._device)

            # Train pl model
            trainer.fit(
                model=self.model,
                train_dataloaders=data.train_loader,
                val_dataloaders=data.valid_loader)

        elif self._type == 'torch':
            raise NotImplementedError
        elif self._type == 'yolo':
            self.model.train(
                data=data, epochs=epochs, device=self._device, batch=batch)

    def export(self, path):
        if self._type == 'pl':
            pass
        elif self._type == 'torch':
            pass
        elif self._type == 'yolo':
            pass
        else:
            raise Exception(f'Model type {self._type} not supported')
        raise NotImplementedError

    def load(self, path):
        # TODO: verify file extension
        if self._type == 'pl':
            tmodel = self.model.model

            self.model = PLFasterRCNN.load_from_checkpoint(
                path, model=tmodel, map_location=torch.device(self._device))
        elif self._type == 'torch':
            model = torch.load(path, map_location=torch.device('cpu'))
            self.model.load_state_dict(model['state_dict'])
        elif self._type == 'yolo':
            # NOTE: Should be fixed
            self.model = YOLO(path, device = self._device)
        else:
            raise Exception(f'Model type {self._type} not supported')

    def predict(self, source):
        """
        Perform prediction using the model

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.

        Returns:
            The prediction results.
        """
        image = utils.read_source(source, self._type)
        if self._type == 'pl':
            self.model.model.eval() # NOTE: I shouldn't do that
            print(image.shape)
            results = self.model(image)
            return {
                'boxes': results['boxes'],
                'scores': results['scores'],
                'lables': results['labels']
            }
        elif self._type == 'torch':
            pass
        elif self._type == 'yolo':
            results = self.model.predict(source, verbose=False)
            boxes = results[0].boxes
            return {
                'boxes': boxes.xyxy, 
                'scores': boxes.conf, 
                'labels': boxes.cls
            }

    def info(self, verbose=True):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)

    @property
    def device(self):
        return self.model.device


# def main():
#     model = Model('fasterrcnn', 2, 'gpu')
#     model.train('./conf/torch.yaml', epochs=10, batch=2, workers=4)


# if __name__ == '__main__':
#     main()
