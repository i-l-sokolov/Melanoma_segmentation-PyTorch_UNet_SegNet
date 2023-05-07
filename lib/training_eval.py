import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd


def restore_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img[0] = img[0] * std[0] + mean[0]
    img[1] = img[1] * std[1] + mean[1]
    img[2] = img[2] * std[2] + mean[2]
    img = Image.fromarray((np.moveaxis(img.numpy(), 0, -1) * 255).astype(np.uint8))
    return img



def fit_epoch(model, optimizer, criterion, train_dataloader, device):
    model.train()

    epoch_loss = 0

    for X, y in train_dataloader:
        X = X.to(device)

        y = y.to(device)

        y_pred = model(X)

        loss = criterion(y_pred, y.float())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_dataloader)


def eval_epoch(model, criterion, metric, valid_dataloader, device):
    model.eval()

    eval_loss = 0
    eval_metric = 0

    with torch.no_grad():
        for X, y in valid_dataloader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            loss = criterion(y_pred, y.float())

            iou = metric(y_pred, y)

            eval_loss += loss.item()
            eval_metric += iou.mean().item()

        eval_loss /= len(valid_dataloader)
        eval_loss /= len(valid_dataloader)

    return eval_loss, eval_metric


def eval_plot(model, valid_dataset, n_pics, title, device, name):
    to_image = transforms.ToPILImage()

    choices = np.random.choice(list(range(len(valid_dataset))), n_pics, replace=False)

    val_images = [valid_dataset[choice][0] for choice in choices]
    ground_truth = [valid_dataset[choice][1] for choice in choices]

    model.eval()
    with torch.no_grad():
        pred_images = model(torch.stack(val_images).to(device)).cpu()

    fig, axs = plt.subplots(3, 6, figsize=(18, 9))
    for i in range(n_pics):
        axs[0, i].imshow(restore_image(val_images[i]))
        axs[0, i].axis('off')
        axs[0, i].set_title('Original Image')

        axs[1, i].imshow(to_image(pred_images[i]))
        axs[1, i].axis('off')
        axs[1, i].set_title('Prediction')

        axs[2, i].imshow(to_image(ground_truth[i].float()))
        axs[2, i].axis('off')
        axs[2, i].set_title('Ground Truth')
    fig.suptitle(title, fontsize=18)
    plt.savefig(f'results/{name}')
    plt.close

def train(model : torch.nn.Module, model_name,
          train_dataloader, valid_dataloader, valid_dataset,
          epochs, optimizer,
          criterion, criterion_name,
          scheduler,
          metric, device, save_model):
    tr, val, met = [],[],[]
    best_iou = 0
    print(f'Training {model_name} with loss function {criterion_name}')
    for epoch in range(epochs):
        train_loss = fit_epoch(model, optimizer,criterion,train_dataloader,device)
        tr.append(train_loss)
        valid_loss, iou = eval_epoch(model,criterion, metric, valid_dataloader, device)
        scheduler.step(iou)
        print(f'{model_name} {criterion_name} epoch {epoch + 1}/{epochs} Tr_loss {train_loss:.4f} Val_loss {valid_loss:.4f} Val_IoU {iou:.4f}')
        if iou > best_iou:
            best_iou = iou
            title = f'{epoch + 1}/{epochs} epoch. Tr_loss {train_loss:.4f} Val_loss {valid_loss:.4f} Val_IoU {iou:.4f}'
            name = model_name + '_' + criterion_name + '_' + '.png'
            eval_plot(model, valid_dataset, 6, title, device, name)
            if save_model:
                torch.save(model.state_dict(), f'results/{model_name}_{criterion_name}.pth')
        val.append(valid_loss)
        met.append(iou)
    df = pd.DataFrame({'train_loss' : tr, 'valid_loss' : val, 'metric' : met, 'model' : model_name,
                      'criterion' : criterion_name, 'epoch' : [x + 1 for x in range(epochs)]})
    return df