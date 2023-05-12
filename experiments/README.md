# Experiments

## BrainWeb Simulations

* Unified registration on/off
* E/O correction on/off

## Brain WMHI Segmentation

```sh
/media/smajjk/Storage/backup/Data/Challenges/WMH/preproc
```

* `N=60` T1w and FLAIR with binary labels for WMHI.

## Brain Tumor Classification

```sh
/media/smajjk/Storage/backup/Data/Challenges/BRATS-raw/nii-raw-unires-validation
```

* GBM: `N=132` T1wc, FLAIR and T2w.
* LGG: `N=106` T1wc, FLAIR and T2w.

## Fausto

Copy data:
```sh
scp -r /media/smajjk/Storage/backup/Data/Challenges/WMH/preproc fausto:/raid/mbrudfors/unires/data/wmhi

scp -r /media/smajjk/Storage/backup/Data/Challenges/BRATS-raw/nii-raw-unires-validation/TCGA-LGG fausto:/raid/mbrudfors/unires/data/tumor/lgg

scp -r /media/smajjk/Storage/backup/Data/Challenges/BRATS-raw/nii-raw-unires-validation/TCGA-GBM fausto:/raid/mbrudfors/unires/data/tumor/gbm
```

Docker build:
```sh
docker build . -t mbrudfors/unires
```

Docker run:
```sh
docker run --gpus '"device=3"' \
    -v /home/mbrudfors/Code/UniRes:/workspace/UniRes \
    -v /raid/mbrudfors/unires:/workspace/data \
    --rm --interactive --tty mbrudfors/unires

cd /workspace/UniRes
pip install -e .
```