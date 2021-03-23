import albumentations as A
import torch
import torch.optim as optim
import torchmetrics as torchmetrics
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # used to show progress bar

from core.dataset.simple_dataset import RoadSegmentationSimpleDataset
# Hyperparameters etc.
from core.model.res_unet import ResUnet
from core.model.res_unet_plus import ResUnetPlusPlus
from core.utils.metrics import BCEDiceLoss

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TODO Is out of memory issue normal for images of size 400x400 or is there an error in the code ?
#  Low BATCH_SIZE and resizing image helps with OOM-Issue IMAGE_HEIGHT, IMAGE_WIDTH 400 ---> define strategies
BATCH_SIZE = 8
IMAGE_HEIGHT_RESIZED = 256
IMAGE_WIDTH_RESIZED = 256

NUM_EPOCHS = 10
NUM_WORKERS = 1
TRAIN_IMG_DIR = "../data/training/images"
TRAIN_MASK_DIR = "../data/training/groundtruth"


def evaluate(model, batch_size, loss_fn, data_loader):
    """
    Evaluate model on validation data.
    """
    model.eval()
    total_loss, total_dice_score = 0., 0.

    # initialize accuracy metric
    accuracy = torchmetrics.Accuracy()
    accuracy.to(DEVICE)

    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image, label = image.to(DEVICE), label.to(DEVICE).float().unsqueeze(1)
            # forward pass
            output = model(image)
            loss = loss_fn(output, label)  # compute loss

            # update loss
            # TODO create loss metric class
            total_loss += loss.item()

            # update accuracy
            accuracy.update(output, label.int())

            # update dice score
            # TODO create dice score metrics class
            preds = torch.sigmoid(output)
            preds = (preds >= 0.5).float()

            total_dice_score += (2 * (preds * label).sum()) / (
                    (preds + label).sum() + 1e-8
            )

    total_loss /= len(data_loader)
    total_accuracy = accuracy.compute()
    total_dice_score /= len(data_loader)

    return total_loss, total_accuracy, total_dice_score


def train_step(model, batch_size, optimizer, loss_fn, data_loader):
    """
    Train model for 1 epoch.
    """
    loop = tqdm(data_loader)

    model.train()
    total_loss, total_dice_score = 0., 0.

    # initialize accuracy metric
    accuracy = torchmetrics.Accuracy()
    accuracy.to(DEVICE)

    for i, (image, label) in enumerate(loop):
        image, label = image.to(DEVICE), label.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()  # zero the parameter gradients

        # forward pass
        output = model(image)
        loss = loss_fn(output, label)  # compute loss

        # backward pass
        loss.backward()
        optimizer.step()  # perform update

        # update loss
        # TODO create loss metric class
        total_loss += loss.item()

        # update accuracy
        accuracy.update(output, label.int())

        # update dice score
        # TODO create dice score metrics class
        preds = torch.sigmoid(output)
        preds = (preds >= 0.5).float()

        total_dice_score += (2 * (preds * label).sum()) / (
                (preds + label).sum() + 1e-8
        )

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    total_loss /= len(data_loader)
    total_accuracy = accuracy.compute()
    total_dice_score /= len(data_loader)

    return total_loss, total_accuracy, total_dice_score


def main():
    transformer = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT_RESIZED, width=IMAGE_WIDTH_RESIZED),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    dataset = RoadSegmentationSimpleDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                            transform=transformer, device=DEVICE)

    TRAIN_SET_SIZE = 0.7
    TRAIN_SET_SIZE_INT = int(len(dataset) * TRAIN_SET_SIZE)

    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       lengths=[TRAIN_SET_SIZE_INT, len(dataset) - TRAIN_SET_SIZE_INT],
                                                       generator=torch.Generator().manual_seed(42))

    print("Train set size:", len(train_set))
    print("Validation set size:", len(val_set))

    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS,
                                                    shuffle=True,
                                                    pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(val_set,
                                                  batch_size=BATCH_SIZE,
                                                  num_workers=NUM_WORKERS,
                                                  shuffle=True,
                                                  pin_memory=True)

    model = ResUnetPlusPlus(3).to(DEVICE)
    loss = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        # train model for one epoch
        train_loss, train_accuracy, train_dice_score = train_step(model=model, loss_fn=loss, optimizer=optimizer,
                                                                  data_loader=train_data_loader, batch_size=BATCH_SIZE)

        # evaluate
        val_loss, val_accuracy, val_dice_score = evaluate(model=model, loss_fn=loss, data_loader=val_data_loader,
                                                          batch_size=BATCH_SIZE)

        print(f"\n[Epoch {epoch}] - Training :   accuracy = {train_accuracy:.5f},"
              f" loss = {train_loss:.5f}, dice score = {train_dice_score:.5f}")

        print(f"......... - Validation : accuracy = {val_accuracy:.5f},"
              f" loss = {val_loss:.5f}, dice score = {val_dice_score:.5f}")


if __name__ == '__main__':
    torch.manual_seed(61274)
    print(f'Using device: {DEVICE}')
    main()
