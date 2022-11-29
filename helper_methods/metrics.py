import torch
import numpy as np
from sklearn.metrics import classification_report
from itertools import product

def compute_confusion_matrix(model, data_loader, device=torch.device('cuda')):

    all_ground_truth, all_predictions = [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            all_ground_truth.extend(labels.to('cpu'))
            all_predictions.extend(predicted_labels.to('cpu'))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
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

def compute_classification_report(test_loader, model, device =torch.device('cpu')):
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