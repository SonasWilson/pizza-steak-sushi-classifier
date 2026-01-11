import torch
from torch import nn, optim
from data_setup import download_data, create_dataloaders
from model_builder import create_model
from engine import train_step, test_step
from torchvision import transforms
from pathlib import Path
from utils import save_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # Paths
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Setup directories
    # train_dir = "../data/pizza_steak_sushi/train"
    # test_dir = "../data/pizza_steak_sushi/test"

    # Create transforms
    # Create a transforms pipeline manually (required for torchvision < 0.13)

    manual_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    #download and prepare data
    download_data(DATA_DIR)
    data_path = DATA_DIR


    train_dataloader, test_dataloader, class_names = create_dataloaders(data_path,
                                                                        transform=manual_transform,
                                                                        batch_size=32)

    # model
    model = create_model(len(class_names)).to(device)

    # define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)

    epochs = 10
    best_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(f"\nEpoch {epoch+1}/{epochs}: ")
        print(f"Train acc: {train_acc:.3f} | "
              f"Train loss: {train_loss:.3f} | "
              f"Test acc: {test_acc:.3f} | "
              f"Test loss: {test_loss:.3f} | "
              )

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()