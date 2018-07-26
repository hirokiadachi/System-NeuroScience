import os
import cv2
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from train import Brief_CNN

## import chainer
import chainer
import chainer.functions as F
from chainer import Variable, serializers

def configure():
    config = argparse.ArgumentParser()
    config.add_argument('--gpu', type=int, default=-1)
    config.add_argument('--label', type=int, default=0)
    config.add_argument('--img', type=str, default='')

    return config.parse_args()

def image_convert(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x *= 0.5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x

def main():
    args = configure()
    label = args.label
    filename = os.path.join('results', 'grad_cam.png')

    mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)
    pil_img = Image.open(args.img)
    pil_img = pil_img.resize((512, 512))
    raw_img = np.asarray(pil_img, dtype=np.float32)
    img = raw_img - mean
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis]
    print('Image Shape: {}'.format(img.shape))

    cnn = Brief_CNN()
    serializers.load_npz('results/model', cnn)

    var_img = Variable(img)
    prediction = cnn(var_img)
    probs = F.softmax(prediction).data[0]

    ## Obtain top5 index
    top5_ind = np.argsort(probs)[::-1][:5]
    prediction.zerograd()
    prediction.grad = np.zeros([1, 5], dtype=np.float32)
    prediction.grad[0, top5_ind[label]] = 1

    prediction.backward(True)
    feature, grad = cnn.cam.data[0], cnn.cam.grad[0]
    cam = np.ones(feature.shape[1:], dtype=np.float32)
    weight = grad.mean((1, 2)) * 1000
    for ind, w in enumerate(weight):
        cam += feature[ind] * w
    cam = np.resize(cam, (512, 512))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    guid = np.maximum(var_img.grad[0], 0)
    guid = guid.transpose(1, 2, 0)
    guided_cam = image_convert(guid * heatmap[:, :, np.newaxis])
    guided_bp = image_convert(guid)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    image = img[0, :].transpose(1, 2, 0)
    image -= np.min(image)
    image = np.minimum(image, 255)
    cam_img = np.float32(heatmap) + np.float32(image)
    cam_img = 255 * cam_img / np.max(cam_img)

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax1.imshow(raw_img[:, :, ::-1].astype(np.uint8))
    ax2.imshow(guided_cam)
    ax3.imshow(heatmap)
    ax4.imshow(cam_img[:, :, ::-1].astype(np.uint8))
    fig.savefig(filename)

if __name__ == '__main__':
    main()
