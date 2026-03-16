import pytorch_lightning as pl

class BaseTrainingModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
