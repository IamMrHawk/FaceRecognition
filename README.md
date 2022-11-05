# Face Recognition


## Installation

Requirements

    • Python 3.6+.
    • macOS or Linux or Windows.

1 : Install the requirements.
```python
$ pip install -r  requirements.txt
```
note : Install PyTorch according to your system environment and Compute Platform from https://pytorch.org/

2 : To generate dataset through videos ( note : save video files in 'VideoDataset' Folder)
```python
$ python3 generate.py
```
3 : To train NN model with generated dataset
```python
$ python3 train.py
```
4 : To run Face Recognition on live cam.
```python
$ python3 main.py
```


