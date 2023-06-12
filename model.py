import torch
import pytorch_lightning as pl
from ultralytics import YOLO
from .dataset import CustomData
from .models.fasterrcnn import TFasterRCNN, PLFasterRCNN
from .models.ssd import TSSD, PLSSD
from .models.retina import TRetina, PLRetina
from . import utils


class Model:
    r"""

    Args
        model (str): type of model
        num_classes (int): including the background
        device (str | list): model device
    """

    def __init__(self, model: str, num_classes=2, device='cpu'):
        self.num_classes = num_classes
        self.model = None
        self._device = device
        self._name = model
        self._type = None

        # Device assertions
        if not (device == 'cpu'):
            if not torch.cuda.is_available():
                raise Exception('GPU training on this device is not supported')
            else:
                if isinstance(device, int):
                    if device >= torch.cuda.device_count():
                        raise Exception(f"Device {device} doesn't exist")
                elif isinstance(device, list):
                    for d in device:
                        if d >= torch.cuda.device_count():
                            raise Exception(f"Device {device} doesn't exist")
                else:
                    raise Exception(f'Device {device} not supported')

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
        
        elif model == 'retina':
            if self._device == 0 or isinstance(device, list):
                self._type = 'pl'
                print("Setting ssd on pytorch lightning")
                tmodel = TRetina(self.num_classes)
                self.model = PLRetina(tmodel.model)
                self.model.to(self._device)
            elif self._device == 'cpu':
                self._type = 'torch'
                self.model = TRetina(self.num_classes)
                self.model.to(self._device)

        elif model == 'yolo':
            self._type = 'yolo'
            self.model = YOLO('yolov8x.yaml')
            self.model.to(self._device)

        else:
            raise Exception(f'Model {model} not supported')

    def train(
            self,
            data: str,
            epochs=10,
            batch=4,
            shuffle=True,
            workers=4,
            out_size=None):
        if self._type == 'pl':
            if out_size:
                data: CustomData = CustomData(
                    data, batch, shuffle, workers, out_size)
            else:
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
        # if self._type == 'pl':
        #     pass
        # elif self._type == 'torch':
        #     pass
        # elif self._type == 'yolo':
        #     pass
        # else:
        #     raise Exception(f'Model type {self._type} not supported')
        raise NotImplementedError

    def load(self, path):
        # TODO: verify file extension
        if self._type == 'pl':
            tmodel = self.model.model
            if self._name == 'fasterrcnn':
                self.model = PLFasterRCNN.load_from_checkpoint(
                    path, model=tmodel, map_location=self.device)
            elif self._name == 'ssd':
                self.model = PLSSD.load_from_checkpoint(
                    path, model=tmodel, map_location=self.device)
            else:
                raise Exception(f'Model name {self._name} not supported')
        elif self._type == 'torch':
            # TODO: torch and gpu ?
            model = torch.load(path, map_location=torch.device('cpu'))
            self.model.load_state_dict(model['state_dict'])
        elif self._type == 'yolo':
            # NOTE: Should be fixed
            self.model = YOLO(path)

            # NOTE: Should be fixed
            self.model.overrides['conf'] = 0.25 # 0.25  # NMS confidence threshold
            self.model.overrides['iou'] = 0.45 # 0.45  # NMS IoU threshold
            self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
            self.model.overrides['max_det'] = 1000  # maximum number of detections per image

            self.model.to(self._device)
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
            self.model.model.eval()  # NOTE: I shouldn't do that
            image = image.unsqueeze(0)  # TODO: Batch problem
            image = image.to(self.device)
            results = self.model(image)
            results = results[0]  # Batch problem
            return {
                'boxes': results['boxes'].cpu().detach(),
                'scores': results['scores'].cpu().detach(),
                'labels': results['labels'].cpu().detach()
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

    def test(self, data, batch=1, workers=4, out_size=None):
        if self._type == 'pl':
            if out_size:
                data: CustomData = CustomData(
                    data, batch, False, workers, out_size)
            else:
                data: CustomData = CustomData(data, batch, False, workers)
            # Print dataset information
            print(data)

            trainer = pl.Trainer(accelerator='gpu')
            results = trainer.test(
                model=self.model, dataloaders=data.test_loader)
            return results[0]
        elif self._type == 'torch':
            raise NotImplementedError
        elif self._type == 'yolo':
            metrics = self.model.val(
                data=data, device=self._device, split = 'test')
            results = {
                'test_map': metrics.box.map,
                'test_map_50': metrics.box.map50,
                'test_map_75': metrics.box.map75
            }
            return results
        else:
            raise Exception(f'Model type {self._type} not supported.')

    @property
    def device(self):
        return self.model.device


# def main():
#     model = Model('fasterrcnn', 2, 'gpu')
#     model.train('./conf/torch.yaml', epochs=10, batch=2, workers=4)


# if __name__ == '__main__':
#     main()
