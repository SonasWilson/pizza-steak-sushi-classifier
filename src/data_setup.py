import zipfile
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path



# get data
DATA_URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"


def download_data(data_path: Path):
#     setup path
    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path/"pizza_steak_sushi.zip"

#     if data doesnt exist download it
    if not zip_path.exists():
        print("Downloading pizza, steak, sushi dataset...")
        request = requests.get(DATA_URL)
        zip_path.write_bytes(request.content)

        # unzipping data
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
            print("Unzipping data...")

        zip_path.unlink()
        print("Download complete.")


# create dataloaders
def create_dataloaders(
data_dir,
        transform: transforms.Compose,
        batch_size: int
):

    # dir
    train_dir = data_dir/"train"
    test_dir = data_dir/"test"

    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # get class name
    class_names = train_data.classes

    # turn images to dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader, class_names