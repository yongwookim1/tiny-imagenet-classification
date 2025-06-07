import argparse

from utils import *


def main(args):
    model_name = args.model
    dataset = args.dataset
    epoch = args.epoch
    batch_size = args.batch_size
    
    device = set_device()
    dataloaders = set_dataloaders(dataset, batch_size)
    model_ft, criterion, _ = set_model(model_name)
    model_ft = model_ft.to(device)

    # Test
    model_ft = load_model(model_ft, f"models/{model_name}/{epoch}_epoch.pt")
    test_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, device=device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--dataset", type=str, default="tiny-224")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    
    main(args)