# System-NeuroScience (Deep Learning vs. Human)
What is the overfitting for human?<br>
To research it , we compare human to deep learning.
main idea : Recognition what clothes own person belong.


# Network Architecture
Constructing brief-Convolutional Neural Network (CNN).<br>
Number of Convolution layer : 5<br>
Number of Fully Connected   : 2<br>
Number of Output node       : 5<br>

* Note *
Please, change network architecture as appropriate.

# Dataset
Took a picture of T-shirts of each member.  Then, these picture use to training dataset.<br>
However, only these are not enough, so that we take a internet.
## How use internet pictures of T-shirts.
Capture the clothes you think want to wear on the internet.<br>
About over 10 clothes are collected for each person.

# How to execute
** Training
```sh
python3 train.py -i [dataset-path]
```
We using chainer(v4.0.0).

** Testing
```sh
python3 train.py --model [pre-train model path] -i [single-image-path] --test
```
Please, input single images network, to output display the probability against the input image.

This network is not so deep and number of training image is only a few.<br>
Thus, training is immediately finish and also recognition rate maybe not good.
So when, change the network architecture and increase the dataset size.

# Requirement
python : 3.x.x<br>
PIL    : 1.1.7<br>
Pillow : 5.1..0<br>
chainer: 4.x.x<br>
tqdm<br>
