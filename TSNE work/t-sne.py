# https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
# https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.utils.data as td
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision import models
import os

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

root_ds_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo"
root_txt_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Model_Stats"
root_model_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Saved_Models/"
root_saved_model_path = "C:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/Damians Stuff/Saved_Models/"

ds_dir_dict = {"EuroSat": root_ds_path+"/LandscapeClassification/Datasets/EuroSat/EuroSAT", 
               "Landuse": root_ds_path+"/LandscapeClassification/Datasets/Land-Use Scene Classification/images",
               "SRSI": root_ds_path+"/LandscapeClassification/Datasets/SRSI RSI CB256/data"}

model_dict = {"VGG": [models.vgg16(weights=None), 4096],
              "ResNet": [models.resnet50(weights=None), 2048],
              "EffNet": [models.efficientnet_b0(weights=None), 2048]}

dict_class_count = {}

for ds in ds_dir_dict:
    data_dir = ds_dir_dict[ds]
    dict_class_count[ds] = count_target_classes(data_dir)
    train_loader, val_loader, test_loader = dataset_loader(data_dir, val_split=.2, test_split=0.2, input_size=[64,64], batch_size=32)

layer_size = model_dict["VGG"][1]
model = model_dict["VGG"][0]  # By default, no pre-trained weights are used.
model.fc = nn.Linear(layer_size, len(dict_class_count["EuroSat"])) # change output parameters to match number of classes
model.load_state_dict(torch.load(root_saved_model_path+"EuroSat_VGG_20.pth"))
model.eval()
model = model.cuda()

# Define your output variable that will hold the output
out = None
# Define a hook function. It sets the global out variable equal to the
# output of the layer to which this hook is attached to.
def hook(module, input, output):
    global out
    out = output
    return None
# Your model layer has a register_forward_hook that does the registering for you
model.classifier[0].register_forward_hook(hook)

# Then you just loop through your dataloader to extract the embeddings
embeddings = np.zeros(shape=(0,layer_size))
labels = np.zeros(shape=(0))
for x,y in iter(test_loader):
    x = x.cuda()
    model(x)
    labels = np.concatenate((labels,y.numpy().ravel()))
    embeddings = np.concatenate([embeddings, out.detach().cpu().numpy()],axis=0)

# test_imgs = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
# test_predictions = []
# test_targets = []
# test_embeddings = torch.zeros((0, 100), dtype=torch.float32)
# for x,y in test_loader:
#     x = x.cuda()
#     embeddings, logits = model(x)
#     preds = torch.argmax(logits, dim=1)
#     test_predictions.extend(preds.detach().cpu().tolist())
#     test_targets.extend(y.detach().cpu().tolist())
#     test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
#     test_imgs = torch.cat((test_imgs, x.detach().cpu()), 0)
# test_imgs = np.array(test_imgs)
# test_embeddings = np.array(test_embeddings)
# test_targets = np.array(test_targets)
# test_predictions = np.array(test_predictions)

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(embeddings)
# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 10
for lab in range(num_categories):
    indices = labels==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.savefig("test_tsne.png")


# plt.show()