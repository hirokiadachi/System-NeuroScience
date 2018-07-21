## System Neuro Science Research
## Deep Learning vs. Human
## Using RGB Images. (Char Datasets and Clothes Datasets)

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

## import chainer sevral library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, optimizers, serializers, cuda, Chain, reporter
from chainer.training import extensions
from chainer.datasets import tuple_dataset

def configuration():
    ## Configure the training parameters by argparse.
    config = argparse.ArgumentParser()
    config.add_argument('--gpu', '-g', type=int, default=0,
                        help='Select GPU Number.')
    config.add_argument('--batch', '-b', type=int, default=2,
                        help='Number of Batch Size.')
    config.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of Total Epoch.')
    config.add_argument('--img', '-i', type=str, default='',
                        help='Directory Path.')
    config.add_argument('--model', '-m', type=str, default=None,
                        help='Pre-Train Model File.')
    config.add_argument('--opt', '-o', type=str, default=None,
                        help='Pre-Train Optimizer File.')
    config.add_argument('--test', '-t', choices=['single', 'multiple'],
                        type=str)
    config.add_argument('--dataset', '-d', choices=['clothes', 'char'],
                        type=str, default='clothes')

    return config.parse_args()


## Make Training Dataset
def make_dataset(dir_path, size=512):
    item_list = sorted(os.listdir(dir_path))[1:]
    
    for index, item in tqdm(enumerate(item_list)):
        item_path = os.path.join(dir_path, item)
        clothes_list = sorted(os.listdir(item_path))[1:]
        for ind, im in enumerate(clothes_list):
            im_path = os.path.join(item_path, im)
            img = Image.open(im_path)
            img_rotate = img.rotate(-90)
            img_resize = img_rotate.resize((size, size))
            img = np.asarray(img_resize, dtype=np.float32)
            img = img.transpose((2, 0, 1))
            img = img[np.newaxis, :, :, :]
            try:
                labels = np.vstack((labels, index))
                datasets = np.vstack((datasets, img))
            except:
                labels = index
                datasets = img

    train = tuple_dataset.TupleDataset(datasets, labels)
    return train

## Make handwriting analysis datasets
def make_dataset_char(dir_path, size=128):
    PATH = []
    PATH.append([os.path.join(dir_path, 'adachi'), 0])
    PATH.append([os.path.join(dir_path, 'inagaki'), 1])
    PATH.append([os.path.join(dir_path, 'maruyama'), 2])
    PATH.append([os.path.join(dir_path, 'onisi'), 3])
    PATH.append([os.path.join(dir_path, 'takatori'), 4])
    all_Data = []
    for item in PATH:
        char_list = os.listdir(item[0])
        char_label = item[1]
        for ind, im in enumerate(char_list):
            im_path = os.path.join(item[0], im)
            img = Image.open(im_path)
            img_resize = img.resize((size, size))
            img = np.asarray(img_resize, dtype=np.float32)
            print(img.shape)
            img = img.transpose((2, 0, 1))
            all_Data.append([img, char_label])

    dataset = np.random.permutation(all_Data)
    IMG = [dataset[i][0] for i in range(len(all_Data))]
    LBL = [dataset[i][1] for i in range(len(all_Data))]
    threshold = np.int32(len(dataset)/8*7)
    train = tuple_dataset.TupleDataset(IMG[0:threshold], 
                                       LBL[0:threshold])
    test  = tuple_dataset.TupleDataset(IMG[threshold:], 
                                       LBL[threshold:])
    return train, test
            

def load_single_image_clothes(image_path, size=512):
    img = Image.open(image_path)
    img_rotate = img.rotate(-90)
    img_resize = img_rotate.resize((size, size))
    img = np.asarray(img_resize, dtype=np.float32)
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    return img

def load_single_image_char(image_path, size=128):
    img = Image.open(image_path)
    img_resize = img.resize((size, size))
    img = np.asarray(img_resize, dtype=np.float32)
    #img = img[:, :, np.newaxis]
    #ones = np.ones((size, size, 3), dtype=np.float32)
    #img = img * ones
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    return img

## Per Layer with training
class Layer(Chain):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1):
        w = chainer.initializers.Normal(1.0)
        super(Layer, self).__init__(
            conv = L.Convolution2D(ch_in, ch_out, ksize=ksize, 
                                   stride=stride, pad=pad, initialW=w),
            bn   = L.BatchNormalization(ch_out),
        )

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = F.leaky_relu(h)

        return h


## Construction Convolutional Neural Net
class Brief_CNN(Chain):
    def __init__(self):
        w = chainer.initializers.Normal(1.0)
        self.cnt = 1
        super(Brief_CNN, self).__init__(
            c0 = Layer(3, 64),
            c1 = Layer(64, 128, ksize=2, stride=2, pad=0),
            c2 = Layer(128, 256),
            c3 = Layer(256, 512, ksize=2, stride=2, pad=0),
            c4 = Layer(512, 512, ksize=2, stride=2, pad=0),
            l0 = L.Linear(None, 128),
            l1 = L.Linear(None, 5)
        )

    def __call__(self, x):

        h = x
        for i in range(5):
            h = self['c%d'%(i)](h)
            if i % 2 == 0:
                self.cam = h
                h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.l0(h))
        out = self.l1(h)

        return out

## Construction Network Updater
class CNNUpdater(chainer.training.StandardUpdater):
    def __init__(self, net_model, **kwargs):
        self.model = net_model
        self.args  = configuration()
        super(CNNUpdater, self).__init__(**kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('optimizer')
        cnn = self.model

        xp = cnn.xp

        batch = self.get_iterator('main').next()
        x = [batch[i][0] for i in range(len(batch))]
        t = [batch[i][1] for i in range(len(batch))]
        x = xp.asarray(x, dtype='f')
        t = xp.asarray(t, dtype='i')
        m = len(x)

        if self.args.test:
            with chainer.using_config('train', False):
                out = cnn(x)
            val = F.softmax(out)
        else:
            out = cnn(x)

        t = xp.ndarray.flatten(t)
        loss = F.softmax_cross_entropy(out, t)
        acc = F.accuracy(out, t)

        if not self.args.test:
            cnn.cleargrads()
            loss.backward()
            optimizer.update()

        reporter.report({'Loss': loss, 'Acc': acc})

def train():
    args = configuration()

    ## Parameter Information Display out
    print('===================================================')
    if args.test:
        print('Evaluation of Network')
        print('Num of Minibatch Size: 1')
        print('Num of Epoch         : 1')
    else:
        print('Training a Network')
        print('Num of Minibatch Size: {}'.format(args.batch))
        print('Num of Epoch         : {}'.format(args.epoch))
    if args.gpu >= 0:
        print('GPU Number           : {}'.format(args.gpu))
    else:
        print('Training with CPU only.')
    print('===================================================')


    ## Set up the Training Network
    model = Brief_CNN()
    if args.model is not None:
        print('Loading Brief CNN model from {}'.format(args.model))
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    ## Set up the Optimizer
    ## Kind of Optimizers are MomentumSGD.
    optimizer = optimizers.MomentumSGD(lr=0.0001)
    optimizer.setup(model)
    if args.opt is not None:
        print('Loading Brief CNN Optimizer from {}'.format(args.opt))
        serializers.load_npz(args.opt, optimizer)

    if args.test == 'single':
        label_name = {0: 'Adachi',
                      1: 'Inagaki',
                      2: 'Maruyama',
                      3: 'Onishi',
                      4: 'Takatori'}

        if args.dataset == 'char':
            test = load_single_image_char(args.img, size=128)
        else:
            test = load_single_image_clothes(args.img, size=512)
        xp = model.xp
        test = xp.asarray(test, dtype=xp.float32)
        with chainer.using_config('train', False):
            val = model(test)
        prob = F.softmax(val).data
        prob = xp.reshape(prob, (prob.shape[1]))
        max_val_ind = xp.ndarray.argmax(prob)
        max_val = prob[max_val_ind] * 100
        print('>>> {} : {} %'.format(label_name[int(max_val_ind)], max_val))

    else:
        if args.dataset == 'clothes':
            train = make_dataset(args.img)
        else:
            train, test = make_dataset_char(args.img, size=128)
        train_iter = iterators.SerialIterator(train, 
                                              batch_size=args.batch)
        test_iter = iterators.SerialIterator(test, batch_size=1)
        log_filename = 'log'
        if args.test == 'multiple':
            ## network evaluate with multiple data
            updater = CNNUpdater(net_model=model,
                                 iterator={'main': test_iter},
                                 optimizer={'optimizer': optimizer},
                                 device=args.gpu)
            trainer = training.Trainer(updater, (1, 'epoch'), 
                                       out='results')
        else:
            ## training network
            updater = CNNUpdater(net_model=model,
                                 iterator={'main': train_iter},
                                 optimizer={'optimizer': optimizer},
                                 device=args.gpu)
            trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                                       out='results')
            trainer.extend(extensions.snapshot_object(model, 'model'),
                           trigger=(10, 'epoch'))
            trainer.extend(extensions.snapshot_object(
                           optimizer, 'optimizer'), 
                           trigger=(10, 'epoch'))


        trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), 
                                            log_name=log_filename))
        trainer.extend(extensions.PrintReport(['epoch', 
                                               'Loss', 
                                               'Acc']))
        trainer.extend(extensions.ProgressBar(update_interval=1))

        trainer.run()


        if not test:
            modelname = 'results/model'
            print('Saving Brief CNN model to {}'.format(modelname))
            serializers.save_npz(modelname, model)

            optname = 'results/optimizer'
            print('Saving Brief CNN optimizer to {}'.format(optname))
            serializers.save_npz(optname, optimizer)

    print('OVER')

if __name__ == '__main__':
    train()
