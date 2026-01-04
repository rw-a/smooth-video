# Smooth Video: A Video Frame Interpolator

> Check out [Smooth Video 2](https://github.com/rw-a/smooth-video-2) for a modern, higher quality model.

## Introduction
 - Smooths video by interpolating frames (e.g. 24fps → 48fps). 
 - Useful for making cinematic slow-mo footage.
 - This is a more user-friendly adaptation of [einanshan's implementation](https://github.com/feinanshan/M2M_VFI) of [Many-to-many Splatting for Efficient Video Frame Interpolation](https://arxiv.org/pdf/2204.03513.pdf%5D(https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Many-to-Many_Splatting_for_Efficient_Video_Frame_Interpolation_CVPR_2022_paper.pdf)) by Ping Hu, Simon Niklaus, Stan Sclaroff, and Kate Saenko. 
 - This method is highly speed optimised for high interpolation factors (e.g. 8x), although quality may still suffer.
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rw-a/smooth-video/blob/main/Smooth_Video.ipynb)

## Requirements
 - Linux
 - Python 3.7
 - NVIDIA GPU + CUDA 10.0

## Installation
1. Clone the repository
```
git clone https://github.com/rw-a/smooth-video
cd video-smoother
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Run the interpolator
```
python main.py input.mp4 output.mp4 --factor 2
```
