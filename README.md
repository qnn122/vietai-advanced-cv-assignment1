# VietAI's Advanced Class in Computer Vision | Assignment No.1 - Retinal Disease Classification
Implementation of Assignment No.1 - Retinal Disease Classification, VietAI's Advanced Class in Computer Vision, 2020

## Quick start
Chech out jupyter notebooks in folder `notebooks` for both tensorflow and pytorch implementations. These implementations are specifically written to run in google's `colab` platform. Users are encouraged to modify the setup section in order to run the notebooks locally.

## Setup
1. Download the dataset (zip file) in this [link](https://drive.google.com/drive/u/0/folders/1hGMGqrK_32sENJfTD5XR3rcfexo0hzRA) (please contanct me if you can't access it)

`Note:` it would take a few minutes to download the dataset (~309 MB).

Unzip the file and put the folder inside `data/`

2. Create a virtual environment and install depedencies:
```bash
pip install -r requirements.txt
```

3. Start training by running
```python
python train.py
```