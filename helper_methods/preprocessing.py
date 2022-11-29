import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td


def load_data(path, test_split, val_split, batch_size, input_size):

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    transform_dict = {'src':  transforms.Compose([transforms.Resize(input_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomAdjustSharpness(0.2),
                                   transforms.RandomAutocontrast(),
                             transforms.ToTensor(), normalize])}

    data = datasets.ImageFolder(root=path, transform=transform_dict["src"])

    val_size = int(len(data) * val_split)
    test_size = int(len(data) * test_split)
    train_size = len(data)- (val_size + test_size)

    train_dataset, test_dataset, val_dataset = td.random_split(data, [train_size, test_size, val_size])

    
    data_loader_train= td.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=True, drop_last = False,
                                     num_workers = 0)
    
    data_loader_test = td.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=True, drop_last = False,
                                     num_workers = 0)
    
    data_loader_val = td.DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=True, drop_last = False,
                                     num_workers = 0)
      
    return data_loader_train, data_loader_test, data_loader_val,data