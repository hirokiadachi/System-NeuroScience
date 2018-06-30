# System-NeuroScience (Deep Learning vs. Human)
# Network Architecture
Constructing brief-Convolutional Neural Network (CNN).<br>
Number of Convolution layer : 5<br>
Number of Fully Connected   : 2<br>
Number of Output node       : 5<br>


# Dataset
Took a picture of T-shirts of each member.  Then, these picture use training dataset.<br>
However, only these are not enough, so that we take a internet.
## How use internet pictures of T-shirts.
Capture the clothes you wanted to wear on the internet.<br>
About 10 clothes are collected for each person.

# How to execute
<h3> Training</h3>
```sh
python3 train.py -i [dataset-path]
```
<br>
<h3>Testing</h3>
```sh
python3 train.py -i [single-image-path] --test 
```
<br>
