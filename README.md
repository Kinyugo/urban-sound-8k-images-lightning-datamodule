# UrbanSound8kImages

This repository contains a PyTorch Lightning DataModule for loading [UrbanSound8kImages](https://www.kaggle.com/gokulrejith/urban-sound-8k-images). The dataset contains 8732 labeled spectrogram sound excerpts (<=4s) of urban sounds from 10 classes: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `enginge_idling`, `gun_shot`, `jackhammer`, `siren`, and `street_music`.

## Description

The actual data loaded has been preprocessed and saved in a [google drive file](https://drive.google.com/file/d/1y6E94oielvL1fiLbXydMSIvA8JYAoOcL/view?usp=sharing). Specifically the zip file from the drive contains with the following directory structure:

```
.
├── test
│   ├── air_conditioner (150)
│   ├── car_horn (65)
│   ├── children_playing (150)
│   ├── dog_bark (150)
│   ├── drilling (150)
│   ├── engine_idling (150)
│   ├── gun_shot (57)
│   ├── jackhammer (150)
│   ├── siren (140)
│   └── street_music (150)
├── train
│   ├── air_conditioner (722)
│   ├── car_horn (309)
│   ├── children_playing (722)
│   ├── dog_bark (722)
│   ├── drilling (722)
│   ├── engine_idling (722)
│   ├── gun_shot (269)
│   ├── jackhammer (722)
│   ├── siren (670)
│   └── street_music (722)
└── val
    ├── air_conditioner (128)
    ├── car_horn (55)
    ├── children_playing (128)
    ├── dog_bark (128)
    ├── drilling (128)
    ├── engine_idling (128)
    ├── gun_shot (48)
    ├── jackhammer (128)
    ├── siren (119)
    └── street_music (128)
```

The total number of files for each dataset is as follows:

```
1322 test
6312 train
1128 val
```

Each image in each of the subfolders is of shape **223 x 217**.

## Example

```python
from src import UrbanSound8kImagesDataModule

dm = UrbanSound8kImagesDataModule(data_dir="./data/", batch_size=256, image_size=(224, 224))

# -----------------------------
# prepare and setup dataloaders
# -----------------------------
dm.prepare_data()
dm.setup()

# ------------
# train
# ------------
for batch in dm.train_dataloader():
    x, y = batch
    print(x.size(), y.size())

# ------------
# validate
# ------------
for batch in dm.val_dataloader():
    x, y = batch
    print(x.size(), y.size())

# ------------
# test
# ------------
for batch in dm.test_dataloader():
    x, y = batch
    print(x.size(), y.size())

```

## Credits

1. To [Gokul Rejithkumar](https://www.kaggle.com/gokulrejith/urban-sound-8k-images), for providing the spectrograms for the original [UrbanSound8k](https://www.kaggle.com/chrisfilo/urbansound8k) dataset.
2. To [Chris Gorgolewski](https://www.kaggle.com/chrisfilo/urbansound8k), for the original UrbanSound8k dataset.
