import torch
import time


def compute_accuracy_and_loss(model, data_loader, criterion, device):

    with torch.no_grad():

        correct_pred= 0
        total = 0
        running_loss = []

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # predictions
            outputs = model(inputs)
            loss = criterion(outputs, labels)  
            _, predicted_labels = torch.max(outputs, 1)

            # loss
            running_loss.append(loss.item())

            # accuracy
            total += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
 

    return correct_pred.float()/total * 100, sum(running_loss)/len(running_loss)

            

def train_model(model, num_epochs, train_loader,
                valid_loader, criterion, 
                optimizer,
                device, early_stopping, file_name,path_to_save_model,logging_interval=100,
                scheduler=None):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list= [], [], [], [], []

    
    for epoch in range(num_epochs):

        batch_loss = []
        batch_acc = []

        model.train()
        for batch_num, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            #forard and back prop
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            # update model parameters
            optimizer.step()

            # loss per iteration
            minibatch_loss_list.append(loss.item())
            batch_loss.append(loss.item())
            
            #Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            if not batch_num % logging_interval:
                batch_acc.append((correct / total) * 100)
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_num:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')
        
        # Train loss per epoch
        train_loss =sum(batch_loss)/len(batch_loss)   

        # Train accuracy per epoch
        train_acc = sum(batch_acc)/len(batch_acc)

        model.eval()
        with torch.no_grad():  

            # #compute validation accuracy and validation loss
            # train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, criterion, device=device)  
            
            #compute validation accuracy and validation loss
            valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, criterion, device=device)            

            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train ACC: {train_acc :.2f}% '
                  f'| Validation ACC: {valid_acc :.2f}%'
                  f'| Train LOSS: {train_loss :.4f}'
                  f'| Validation LOSS: {valid_loss :.4f}')


            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc.item())

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            # early stopping
            early_stopping('%.4f'%train_loss, '%.4f'%valid_loss)
            if early_stopping.early_stop:
              #saving the best model
              print("Early Stopping --- Saving the final model")
              torch.save(model, path_to_save_model+file_name+'_FINAL_MODEL_WEIGHTS.pth')
              print("Final Model Saved. Training Complete!")
              break #stopping the training

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
        
        scheduler.step()

        if (epoch+1)%8 == 0:
            print("Saving intermediate model weights ")
            torch.save(model, path_to_save_model + file_name+"_" + str(epoch) +'_Intermdiate_MODEL_WEIGHTS.pth')

        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    print("Saving FINAL model weights ")
    torch.save(model, path_to_save_model + file_name+'_FINAL_MODEL_WEIGHTS.pth')


    return minibatch_loss_list, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list