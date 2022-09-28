# Video Frame Interpolator

## Introduction
 - Smooths video by interpolating frames (e.g. 24fps → 48fps). 
 - This is a user-friendlier version of [einanshan's implementation](https://github.com/feinanshan/M2M_VFI) of [Many-to-many Splatting for Efficient Video Frame Interpolation](https://arxiv.org/pdf/2204.03513.pdf%5D(https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Many-to-Many_Splatting_for_Efficient_Video_Frame_Interpolation_CVPR_2022_paper.pdf)) by Ping Hu, Simon Niklaus, Stan Sclaroff, and Kate Saenko. 

## Requirements
 - Linux
 - Python 3.7
 - NVIDIA GPU + CUDA 10.0

## Installation
1. Clone the repository
```
git clone https://github.com/rw-a/video-smoother
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
