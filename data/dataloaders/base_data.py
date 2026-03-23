import lightning as l


class BaseDataModule(l.LightningDataModule):
    def __init__(self, train_loader, valid_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # self.test_mean = test_loader.dataset.mean
        # self.test_std = test_loader.dataset.std
        # self.data_name = test_loader.dataset.data_name
        self.data_name = test_loader.dataset.data_name
        # print(self.data_name)
        
    def train_dataloader(self):
        if self.train_loader is None:
            raise ValueError("Train loader is not set")
        return self.train_loader

    def val_dataloader(self):
        if self.valid_loader is None:
            raise ValueError("Validation loader is not set")
        return self.valid_loader

    def test_dataloader(self):
        if self.test_loader is None:
            raise ValueError("Test loader is not set")
        return self.test_loader