import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import numpy as np

args = sys.argv
item = 'results/log'
with open(item, 'r') as f:
    plot_data = json.load(f)
loss = [plot_data[i]['Loss'] for i in range(len(plot_data))]
acc = [plot_data[i]['Acc'] for i in range(len(plot_data))]

plt.xkcd()   ## Make like hand writing glaph at this line
plt.figure()
plt.gca().yaxis.set_tick_params(direction='in')
plt.gca().xaxis.set_tick_params(direction='in')
tag_loss = 'loss'
tag_acc = 'accuracy'
plt.plot(loss, label=tag_loss)
plt.plot(acc, label=tag_acc)
plt.legend()
plt.xlabel('epoch')
plt.savefig('Pred_Loss.jpg')
