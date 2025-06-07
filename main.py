import argparse

from utils import *


def main(args):
    model_name = args.model
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    
    set_seed()
    
    device = set_device()
    dataloaders = set_dataloaders(dataset, batch_size)
    model_ft, criterion, optimizer_ft = set_model(model_name)
    model_ft = model_ft.to(device)

    # Train
    best_epoch = train_model(
        output_path=model_name,
        model=model_ft,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer_ft,
        device=device,
        num_epochs=epochs,
    )

    # Test
    model_ft = load_model(model_ft, f"models/{model_name}/{best_epoch}_epoch.pt")
    test_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, device=device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="wideresnet")
    parser.add_argument("--dataset", type=str, default="tiny-224")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    
    main(args)