import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_data_loader(data_loader, gridDims):
    plt.style.use('seaborn-white')
    fig, axes = plt.subplots(nrows=gridDims[0], ncols=gridDims[1], figsize=(10,10))
    dataiter = iter(data_loader)
    for i in range(gridDims[0]):
        for j in range(gridDims[1]):
            images, _ = dataiter.next()
            axes[i, j].imshow(np.transpose(images[0].numpy(), (1, 2, 0)))


def plot_training_batch_loss(minibatch_loss_list, num_epochs, iter_per_epoch, averaging_iterations=200):

    plt.figure(figsize=(7,7))
    plt.subplot(1, 1, 1)
    plt.plot(range(len(minibatch_loss_list)),
             (minibatch_loss_list), label='Minibatch Loss')

    plt.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'), label='Running Average')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Minibatch Train Loss")
    plt.legend()
    plt.tight_layout()


def plot_accuracy(train_acc_list, valid_acc_list):

    num_epochs = len(train_acc_list)
    plt.figure(figsize=(7,7))
    plt.plot(np.arange(1, num_epochs+1),
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Train VS Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_loss(train_loss_list, valid_loss_list):

    num_epochs = len(train_loss_list)
    plt.figure(figsize=(7,7))
    plt.plot(np.arange(1, num_epochs+1),
             train_loss_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_loss_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title("Train VS Validation Loss")
    plt.tight_layout()


def show_examples(model, data_loader, unnormalizer=None, class_dict=None, nrows=3, ncols=4):
    
        
    for _, (inputs, labels) in enumerate(data_loader):

        with torch.no_grad():
            inputs = inputs
            labels = labels
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
        break

    _, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True)
    
    if unnormalizer is not None:
        for idx in range(inputs.shape[0]):
            inputs[idx] = unnormalizer(inputs[idx])
            
    unnormalised_image = np.transpose(inputs, axes=(0, 2, 3, 1))

    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(unnormalised_image[idx])
        if class_dict is not None:
            ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                f'\nT: {class_dict[labels[idx].item()]}')
        else:
            ax.title.set_text(f'P: {predictions[idx]} | T: {labels[idx]}')
        ax.axison = False

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix,class_names=None):

    total_samples = confusion_matrix.sum(axis=1)[:, np.newaxis]
    normed_confusion_matrix = confusion_matrix.astype('float') / total_samples



    figsize = (len(confusion_matrix)*1.25, len(confusion_matrix)*1.25)
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    matrixshow = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    fig.colorbar(matrixshow)


    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            block_text = ""
            block_text += format(confusion_matrix[i, j], 'd')
            ax.text(x=j, y=i, s=block_text, va='center', 
                    color="white" if normed_confusion_matrix[i, j] > 0.5 else "black")
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    
    return fig, ax