import torch
from typing import Tuple

# for training
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:

    # model in train mode
    model.train()

    # setup train loss and accuracy
    train_loss, train_acc = 0, 0

    # loop through dataloader batches of data
    for X,y in dataloader:
        # set to target device
        X,y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # adjust to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
) -> Tuple[float,float]:

    # model in eval mode
    model.eval()

    # set test loss and acc values
    test_loss, test_acc =0, 0

    # turn on inference mode
    with torch.inference_mode():
        # loop through dataloader
        for X, y in dataloader:
            # send data to target device
            X, y = X.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(X)

            # calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

        # adjust to get average loss and accuracy per batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        return test_loss, test_acc

