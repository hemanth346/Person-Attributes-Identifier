import matplotlib.pyplot as plt
import numpy as np
import uuid


# labels = [key for key, _ in targets.items()]
# labels


def plot_feature_history(model_history, feature, save=False):
    """
    Plot accuracy and loss plot for a feature from history object
    """
    acc = feature + '_acc'
    val_acc = 'val_' + feature + '_acc'

    loss = feature + '_loss'
    val_loss = 'val_' + feature + '_loss'
    # visualize
    f,ax = plt.subplots(1,2, figsize=(12,6))

    # summarize history for loss
    ax[0].plot(model_history.history[loss], label='train_loss')
    ax[0].plot(model_history.history[val_loss], label='val_loss')
    ax[0].set_title(feature+'_Loss')
    ax[0].set_xlabel('epochs')
    ax[0].legend(loc='upper right')

    # summarize history for accuracy
    ax[1].plot(model_history.history[acc], label='train_acc')
    ax[1].plot(model_history.history[val_acc], label='val_acc')
    ax[1].set_title(feature+'_Accuracy')
    ax[1].set_xlabel('epochs')
    ax[1].legend(loc='upper left')
    if save:
        plt.savefig(feature+'_'+str(uuid.uuid4())+'.jpg')
    plt.show()


# for label in labels:
#     plot_training_epochs(model_info, label)
    
 

def plot_cumulative(model_history, save=False):
    # visualize
    f , ax = plt.subplots(1,2, figsize=(20,10))
    xvalues = model_history.epoch 
    # epochs = 0
    for feature in model_history.history.keys():
        # get all accuracy and losses from the history object
        values = model_history.history[feature]
        # if not epochs:
        #     epochs = len(values)
        #     xvalues = np.arange(epochs)

        if feature.endswith('loss'):
            ax[0].plot(xvalues, values, label=feature)

        elif feature.endswith('acc'):
            ax[1].plot(xvalues, values, label=feature)
    ax[0].set_title("Loss curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend(loc='upper right')

    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend(loc='upper left')
    if save:
        plt.savefig('model_'+str(uuid.uuid4())+'.jpg')
    plt.show()