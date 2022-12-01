import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

def per_dataset(root_path, datasets, super_columns, label_dict):
    for ds in datasets:
        for c in super_columns:
            columns = ["Epoch", c]
            label_dict = {"MeanAcc":"Mean Accuracy", "MeanLoss":"Mean Loss", "ValAcc":"Validation Accuracy", "ValLoss":"Validation Loss"}
            df_VGG = pd.read_csv(root_path+"Stats/"+ds+"_VGG.csv", usecols=columns)
            df_ResNet = pd.read_csv(root_path+"Stats/"+ds+"_ResNet.csv", usecols=columns)
            df_EffNet = pd.read_csv(root_path+"Stats/"+ds+"_EffNet.csv", usecols=columns)
    
            plt.xlabel(columns[0])
            plt.ylabel(label_dict[c])
            plt.plot(df_VGG.Epoch, df_VGG[c], label = "VGG")
            plt.plot(df_ResNet.Epoch, df_ResNet[c], label = "ResNet")
            plt.plot(df_EffNet.Epoch, df_EffNet[c], label = "EffNet")
            plt.legend()
            plt.title(ds+": "+label_dict[columns[1]])
            plt.savefig(root_path+"Plots/Per Dataset/"+ds+"_"+c+".png")
            plt.close()

def per_model(root_path, models, super_columns, label_dict):
    for m in models:
        for c in super_columns:
            columns = ["Epoch", c]
            df_euro = pd.read_csv(root_path+"Stats/EuroSat_"+m+".csv", usecols=columns)
            df_landuse = pd.read_csv(root_path+"Stats/Landuse_"+m+".csv", usecols=columns)
            df_srsi = pd.read_csv(root_path+"Stats/SRSI_"+m+".csv", usecols=columns)

            plt.xlabel(columns[0])
            plt.ylabel(label_dict[c])
            plt.plot(df_euro.Epoch, df_euro[c], label = "EuroSat")
            plt.plot(df_landuse.Epoch, df_landuse[c], label = "Landuse")
            plt.plot(df_srsi.Epoch, df_srsi[c], label = "SRSI")
            plt.legend()
            plt.title(m+": "+label_dict[columns[1]])
            plt.savefig(root_path+"Plots/Per Model/"+m+"_"+c+".png")
            plt.close()
def trainVsVal_Acc(root_path, models, label_dict):
    for ds in datasets:
        for m in models:
            columns = ["Epoch", "MeanAcc", "ValAcc"]
            df = pd.read_csv(root_path+"Stats/"+ds+"_"+m+".csv", usecols=columns)
    
            plt.xlabel(columns[0])
            plt.plot(df.Epoch, df.MeanAcc, label = "Train Accuracy")
            plt.plot(df.Epoch, df.ValAcc, label = "Validation Accuracy")
            plt.legend()
            plt.title(ds+" - "+m+": Train Accuracy vs "+label_dict[columns[2]])
            plt.savefig(root_path+"Plots/Per Model/Train vs. Validation/Accuracy/"+ds+"_"+m+"_train_acc.png")
            plt.close()

def trainVsVal_Loss(root_path, models, label_dict):
    for ds in datasets:
        for m in models:
            columns = ["Epoch", "MeanLoss", "ValLoss"]
            df = pd.read_csv(root_path+"Stats/"+ds+"_"+m+".csv", usecols=columns)

            plt.xlabel(columns[0])
            plt.plot(df.Epoch, df.MeanLoss, label = "Train Loss")
            plt.plot(df.Epoch, df.ValLoss, label = "Validation Loss")
            plt.legend()
            plt.title(ds+" - "+m+": Train Loss vs "+label_dict[columns[2]])
            plt.savefig(root_path+"Plots/Per Model/Train vs. Validation/Loss/"+ds+"_"+m+"_train_loss.png")
            plt.close()

# ---------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------

root_path = "c:/Users/Damian/Documents/School/Fall 2022 - Grad School/Comp 6721/Project/Repo/My Work/Model_Stats/"
datasets = ["EuroSat", "Landuse", "SRSI"]
models = ["VGG", "ResNet", "EffNet"]
super_columns = ["MeanAcc", "MeanLoss", "ValAcc", "ValLoss"]
label_dict = {"MeanAcc":"Mean Accuracy", "MeanLoss":"Mean Loss", "ValAcc":"Validation Accuracy", "ValLoss":"Validation Loss"}

per_dataset(root_path, datasets, super_columns, label_dict)
per_model(root_path, models, super_columns, label_dict)
trainVsVal_Acc(root_path, models, label_dict)
trainVsVal_Loss(root_path, models, label_dict)