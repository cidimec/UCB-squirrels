# DATASET UTILS

In this section we define some helping notebooks to process the data acquired by the OAKs.

- `OAKpreprocessing.ipynb` Converts the coded recordings from the OAKs to `.mp4` format and arranges these recordings.
- `OAKarrange.ipynb` Splits the OAK recordings, performs background subtraction, creates gait representations, and trains a classificator model.
- `ImageSeg.ipynb` It performs image augmentation using a labeled dataset. Then, it trains a U-NET for silhouette segmentation.
- `UCB_arrange.ipynb` Splits the UCB dataset recordings, performs background subtraction, creates gait representations, and trains a classificator model.

