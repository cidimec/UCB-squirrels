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
### Save video files in encoded format
Go to UCB-squirrels/dataRecording and run `python video_encoding.py`. The encoded videos will be saved in a new directory in `.h265` and `.h264` format.
#### Optional arguments:
```
-h, --help            show this help message and exit
-dir DIRECTORY, --directory DIRECTORY
                      Directory where the videos are stored
-rgb C_RES, --c_res C_RES
                      RGB camera resolution height: (1920x)1080, (3840x)2160 or (4056x)3040. Default: 1080
-mono M_RES, --m_res M_RES
                      Mono cam resolution height: (1280x)720, (1280x)800 or (640x)400. Default: 720
-f FPS, --fps FPS     Defines the output frame rate. Default: 30
-sec SEC, --sec SEC   Defines the time (seconds) that the videos will be recorded. Default: 30
-v VERB, --verb VERB  Defines the output verbosity
```

## Authors
* Israel Tiñini Alvarez [email](mailto:ir.tinini@acad.ucb.edu.bo) [Linkedin](https://www.linkedin.com/in/isratial/)
## Cite This Project

If you use this resource in your project or wish to refer to this repository `https://github.com/cidimec/UCB-squirrels`

<!-- ```bash
@misc{relabeller,
  author =       {Israel Tiñini and Benjamin Pinaya},
  title =        {},
  howpublished = {},
  year =         {}
}
``` -->
