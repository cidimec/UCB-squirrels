<h4 align="center">
    DATA RECORDING
</h4>


<div align="center">
  <a href="#Description"><b>Description</b></a> |
  <a href="#Features"><b>Features</b></a> |
  <a href="#Requirements"><b>Requirements</b></a> |
  <a href="#Installation"><b>Installation</b></a> |
  <a href="#usage"><b>Usage</b></a>
</div>


## Description
The files in this directory are designed to record video data in h265 and h264 format using the OAK-D from OpenCV. It is written in Python3 and uses depthai[GEN2].

<!-- <div align="center">
  <img src=".readme/preview2.gif" width="70%">
  <p align="center" ><i>Example of instance segmentation.</i></p>
</div> -->


## Features
- It saves the videos in h265 and h264 format.
- The RGB camera saves the videos in 1080P resolution
- The mono cameras are saved in 720P resolution

## Requirements
- Python3
- OpenCV > 3.0
- depthai > 2.0


## Installation
### Anaconda
### Ubuntu
```
In the depth environment installed with depthai

# Clone the repository and install the requirements
git clone https://github.com/luxonis/depthai-python.git
cd depthai-python/examples/
./install_requirements.py

# (Only for raspberry3 or lower)
cd ../
git fetch --all
git checkout origin/rpi_crash_mitigation
cd examples
./install_requirements.py
```
### Windows


## Usage
Go to UCB-squirrels/dataRecording and run `python both_encoding_max_limit.py`

The encoded videos will be saved in a new directory named with the current time in `.h265` and `.h264` format.

## Authors
* [Israel Tiñini Alvarez](mailto:i.tinini.a@gmail.com)

<!-- ## Cite This Project

If you use this project in your research or wish to refer to the baseline results published in the README, please use the following BibTeX entry.

```bash
@misc{relabeller,
  author =       {Israel Tiñini and Benjamin Pinaya},
  title =        {},
  howpublished = {},
  year =         {}
}
``` -->
