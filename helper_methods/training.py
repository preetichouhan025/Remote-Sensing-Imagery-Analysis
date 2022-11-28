import torch
import time


def compute_accuracy_and_loss(model, data_loader, criterion, device):

    with torch.no_grad():

        correct_pred= 0
        total = 0
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(data_loader):


            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)  
            _, predicted_labels = torch.max(outputs, 1)

            total += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
            running_loss = running_loss + loss.item() 

        mean_loss = running_loss/ total

    return correct_pred.float()/total * 100, mean_loss

            

def train_model(model, num_epochs, train_loader,
                valid_loader, criterion, 
                optimizer,
                device, early_stopping, path_to_save_model,logging_interval=100,
                scheduler=None,threshold = 85,
                scheduler_on='valid_acc'):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list= [], [], [], [], []

    
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            #forard and back prop
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            # update model parameters
            optimizer.step()

            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():  
            #compute train accuracy and train loss
            train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, criterion, device=device)
            #compute validation accuracy and validation loss
            valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, criterion,device=device)

            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train ACC: {train_acc :.2f}% '
                  f'| Validation ACC: {valid_acc :.2f}%'
                  f'| Train LOSS: {train_loss :.4f}% '
                  f'| Validation LOSS: {valid_loss :.4f}%')


            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

            train_loss_list.append(train_loss.item())
            valid_loss_list.append(valid_loss.item())

            # early stopping
            early_stopping('%.4f'%train_loss, '%.4f'%valid_loss)
            if early_stopping.early_stop:
              #saving the best model
              print("Early Stopping --- Saving the final model")
              torch.save(model.state_dict(), path_to_save_model+'FINAL_MODEL_WEIGHTS.pt')
              print("Final Model Saved. Training Complete!")
              break #stopping the training

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

        if epoch%3==0:
            print("Saving intermediate model weights ")
            torch.save(model.state_dict(), path_to_save_model+'Intermdiate_MODEL_WEIGHTS.pt')

        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    print("Saving FINAL model weights ")
    torch.save(model.state_dict(), path_to_save_model+'FINAL_MODEL_WEIGHTS.pt')


    return minibatch_loss_list, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list