import matplotlib.pyplot as plt
import numpy as np

epochs = 20

def draw_plot(name, index, span, legend_title=None):
    if legend_title is None:
        legend_title = name
    d = np.loadtxt(name+'_results.csv').astype('float')

    fig, ax = plt.subplots()
    for i in range(span):
        r = i*epochs
        ax.plot(d[r:r+epochs, 0], d[r:r+epochs, 1], label=str(d[r,index]))
    ax.legend(loc='upper right', title=legend_title)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title('Training loss: '+name)
    fig.savefig(name+'_trainingloss.png')

    fig, ax = plt.subplots()
    for i in range(span):
        r = i*epochs
        ax.plot(d[r:r+epochs, 0], d[r:r+epochs, 2], label=str(d[r,index]))
    ax.legend(loc='upper right', title=legend_title)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title('Validation loss: '+name)
    fig.savefig(name+'_validationloss.png')

    fig, ax = plt.subplots()
    for i in range(span):
        r = i*epochs
        ax.plot(d[r:r+epochs, 0], d[r:r+epochs, 3], label=str(d[r,index]))
    ax.legend(loc='lower right', title=legend_title)
    ax.set_ylabel('accuracy %')
    ax.set_xlabel('epoch')
    ax.set_title('Accuracy: '+name)
    fig.savefig(name+'_accuracy.png')

#draw_plot('momentum', 5, 5)
#draw_plot('decay', 6, 6)
#draw_plot('sigmoid', 4, 4, legend_title='learning rate')
#draw_plot('relu', 4, 4, legend_title='learning rate')
#draw_plot('dropout', 7, 5)

draw_plot('twolayer_lr', 4, 3) 
draw_plot('twolayer_momentum', 5, 3) 
