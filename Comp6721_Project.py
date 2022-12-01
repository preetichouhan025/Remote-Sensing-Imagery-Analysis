import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from sklearn.model_selection import KFold
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import image
import time
import os, random
import pandas as pd
from torchvision.io import read_image
import datetime
import torchmetrics as tm
from sklearn.metrics import classification_report
from itertools import product

def count_target_classes(root_dir):
    '''return count of number of images per class'''
    target_classes = {}
    for folder in sorted(os.listdir(root_dir)):
        data_path_for_image_folder = root_dir+ '/'+str(folder) + '/'
        target_classes[str(folder)] = len([image_filename for image_filename in sorted(os.listdir(data_path_for_image_folder))])
    return target_classes

def dataset_loader(path, val_split, test_split, input_size, batch_size, shuffle_test=False):
    
    transform_dict = {'src':  transforms.Compose([transforms.Resize(input_size), 
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomAdjustSharpness(0.2),
                                                transforms.RandomAutocontrast(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])}

    data = datasets.ImageFolder(root=path, transform=transform_dict["src"])

    test_size = int(len(data) * test_split)
    train_size_temp = len(data)- test_size

    train_dataset_temp, test_dataset = td.random_split(data, [train_size_temp, test_size])

    val_size = int(len(train_dataset_temp) * val_split)
    train_size = len(train_dataset_temp) - val_size

    # split validate set from train set: https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    train_dataset, val_dataset = td.random_split(train_dataset_temp, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)

    return train_loader, val_loader, test_loader

def train(num_epochs, ds, modelName, model, train_loader, criterion, optimizer, val_loader):
    root_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    model.to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    start = datetime.datetime.now()
    t = open(root_path+"/Model_Stats/Stats/"+ds+"_"+modelName+'_time.txt', 'w')
    t.write("start training: "+str(start)+"\n\n")

    f = open(root_path+"/Model_Stats/Stats/"+ds+"_"+modelName+'.csv', 'w')
    f.write("Epoch,MeanAcc,MeanLoss,LR,ValAcc,ValLoss\n")
    
    total_steps = len(train_loader)
    e_lossLst = []
    e_accLst = []

    # TRAIN LOOP
    for epoch in range(num_epochs):
        b_lossLst = []
        b_accLst = []
        for i, data in enumerate(train_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            b_lossLst.append(loss)
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _,predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            b_accLst.append((correct / total) * 100)

        # EPOCH STATS
        meanAcc = sum(b_accLst) / len(b_accLst)
        meanLoss = sum(b_lossLst) / len(b_lossLst)
        # e_lossLst.append(meanAcc)
        # e_accLst.append(meanLoss)

        # VALIDATION LOOP - https://towardsdatascience.com/how-to-cook-neural-nets-with-pytorch-7954c1e62e16
        model.eval()
        valid_loss = 0
        vAccLst = []
        vLossLst = []
        # turn off gradients for validation
        with torch.no_grad():
            for j, vData in enumerate(val_loader):
                vImages, vLabels = vData[0].to(device), vData[1].to(device)
                # forward pass
                vOutput = model(vImages)
                # validation batch loss
                vLoss = criterion(vOutput, vLabels) 
                vLossLst.append(vLoss)
                # accumulate the valid_loss
                valid_loss += loss.item()
                vTotal = vLabels.size(0)
                v_,vPredicted = torch.max(vOutput.data, 1)
                vCorrect = (vPredicted == vLabels).sum().item()
                vAccLst.append((vCorrect / vTotal) * 100)
        vMeanAcc = sum(vAccLst) / len(vAccLst)
        vMeanLoss = sum(vLossLst) / len(vLossLst)
        f.write('{},{:.2f},{:.2f},{},{:.2f},{:.2f}\n'.format(epoch + 1, meanAcc, meanLoss, scheduler.get_last_lr(),vMeanAcc,vMeanLoss))
        scheduler.step()

        # Save model every fifth epoch for safety
        if((epoch+1)%5 == 0):
            torch.save(model.state_dict(), root_path+'/Saved_Models/'+ds+"_"+m+"_"+str(epoch+1)+".pth")

        # Early Stopping
        # if(abs(meanLoss-vMeanLoss)<.0001):
        #     torch.save(model.state_dict(), root_path+'/Saved_Models/In_Progress/'+ds+"_"+m+"_"+str(epoch+1)+".pth")
        #     break

    end = datetime.datetime.now()
    elapsed = ((end-start).seconds)/60
    t.write('end training: '+str(end)+"\n\nelapsed: "+str(elapsed)+" minutes\n")

    return model

def compute_classification_report(test_loader, model, device =torch.device('cuda')):
    target_list = []
    predictions_list = []

    for batch, (images, targets) in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device)
            logits = model(images)
            _, predicted_labels = torch.max(logits, dim=1)

            target_list.extend(targets.tolist())
            predictions_list.extend(predicted_labels.tolist())


    return classification_report(predictions_list, target_list,  zero_division=0)

def compute_confusion_matrix(model, data_loader, device=torch.device('cuda')):

    all_ground_truth, all_predictions = [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            all_ground_truth.extend(labels.to('cuda'))
            all_predictions.extend(predicted_labels.to('cuda'))

    all_predictions = torch.tensor(all_predictions, device = "cpu")
    all_predictions = np.array(all_predictions)

    all_ground_truth = torch.tensor(all_ground_truth, device = "cpu")
    all_ground_truth = np.array(all_ground_truth)
        
    class_labels = np.unique(np.concatenate((all_ground_truth, all_predictions)))

    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])

    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_ground_truth, all_predictions))

    for combination in product(class_labels, repeat=2):
        lst.append(z.count(combination))

    matrix = np.asarray(lst)[:, None].reshape(n_labels, n_labels)

    return matrix

def plot_confusion_matrix(ds, modelName, confusion_matrix,class_names=None):

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

    root_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Model_Stats"
    plt.savefig(root_path+"/Plots/"+ds+"_"+modelName+"_cm.png")

    return fig, ax

def test(ds, modelName, model, test_loader, label_names):
    root_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Model_Stats"
    f = open(root_path+"/Stats/"+ds+"_"+modelName+'TEST.txt', 'w')
    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Test Device: {}".format(device))
    model.to(device)
    model.eval() 

    with torch.no_grad(): 
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        f.write(compute_classification_report(test_loader, model, device=torch.device('cuda') ))
        confusion_matrix = compute_confusion_matrix(model = model, data_loader = test_loader, device=torch.device('cuda'))
        labels = torch.tensor(labels, device = "cpu")
        plot_confusion_matrix(ds, modelName, confusion_matrix, class_names = label_names)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
root_ds_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo"
root_txt_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Model_Stats"
root_model_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Saved_Models/"

ds_dir_dict = {"EuroSat": root_ds_path+"/LandscapeClassification/Datasets/EuroSat/EuroSAT", 
               "Landuse": root_ds_path+"/LandscapeClassification/Datasets/Land-Use Scene Classification/images",
               "SRSI": root_ds_path+"/LandscapeClassification/Datasets/SRSI RSI CB256/data"}

model_dict = {"VGG": [models.vgg16(weights=None), 4096],
              "ResNet": [models.resnet50(weights=None), 2048],
              "EffNet": [models.efficientnet_b0(weights=None), 2048]}

# FOR TESTING
# ds_dir_dict = {"EuroSat": "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/LandscapeClassification/Datasets/EuroSat/EuroSAT"}
# model_dict = {"VGG": [models.vgg16(weights=None), 4096]}

for ds in ds_dir_dict:
    data_dir = ds_dir_dict[ds]
    dict_class_count = count_target_classes(data_dir)
    train_loader, val_loader, test_loader = dataset_loader(data_dir, val_split=.2, test_split=0.2, input_size=[64,64], batch_size=32)

    # LOAD MODEL AND TEST LOOP
    # for m, val in model_dict.items():
    #     model = val[0]  # By default, no pre-trained weights are used.
    #     model.fc = nn.Linear(val[1], len(dict_class_count)) # change output parameters to match number of classes
    #     model.load_state_dict(torch.load(root_model_path+"In_Progress/EuroSat_VGG_20.pth"))
    #     test(ds, m, model, test_loader, dict_class_count.keys())

    for m, val in model_dict.items():
        model = val[0]  # By default, no pre-trained weights are used.
        model.fc = nn.Linear(val[1], len(dict_class_count)) # change output parameters to match number of classes

        # define the loss function
        criterion = nn.CrossEntropyLoss()
        #Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        num_epochs = 20
        trained_model = train(num_epochs, ds, m, model, train_loader, criterion, optimizer, val_loader)
        test(ds, m, trained_model, test_loader, dict_class_count.keys())


