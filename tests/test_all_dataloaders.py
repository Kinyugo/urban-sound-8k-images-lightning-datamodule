from src import UrbanSound8kImagesDataModule


def test_all_dataloaders():
    dm = UrbanSound8kImagesDataModule(data_dir="./data/", batch_size=256)
    dm.prepare_data()
    dm.setup()

    assert len(dm.train_dataloader()) == 25
    assert len(dm.val_dataloader()) == 5
    assert len(dm.test_dataloader()) == 5
