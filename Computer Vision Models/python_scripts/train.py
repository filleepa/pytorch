
import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

def main():
    # Setting up hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # setup agnostic code for target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])

    # Create dataloaders with help from data_setup.py 
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py 
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # set loss and optimizer functions
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # start training with engine.py 
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    # Save the model using utils.py 
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="tinyvgg_image_classification_model.pth")

if __name__ == "__main__":
    print("Starting train.py...", flush=True)
    main()
