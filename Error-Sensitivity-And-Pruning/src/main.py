import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms 
from torch.otpim.lr_scheduler import StepLR 
import argparse 
import random 
import time 
from typing import Any, Tuple 
import components
from enums import modelArchEnum, metricEnum, metadataEnum, datasetEnum
from modules import ConvolutionalNN 
from components import corruption



def main() -> None: 
    parser = argparse.ArgumentParser(description = "Bash file args")

    parser.add_argument('--all_corruption', nargs = '*', help = 'corruption_rates_arr')
    parser.add_argument('--model_architecture', type = str, required = True, help = "model type")
    parser.add_argument('--base_folder', type = str, help = 'base path to store outputs')
    parser.add_argument('--dataset', type = str, required = True, help = 'dataset trained for')
    parser.add_argument('--jobid', type = str, required = True, help = 'jobid')
    parser.add_argument('--epochs', type = int, required = True, help = 'num epochs to train for')
    parser.add_argument('--gamma', type = float, default = 0.7, metavar = 'M', help = "learning rate step gamma")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()

    print("Booted an Running")

    useCuda = not args.no_cuda and torch.cuda.is_available()
    useMps = not args.no_mps and torch.backends.mps_is_available()



    device: torch.device = torch.device("cuda" if useCuda else "mps" if useMps else "cpu")
    print(f"using device: {device.type}")

    trainKwargs: dict = {"batch_size": args.batch_size}
    testKwargs: dict = {"batch_size": args.test_batch_size}

    dataset_train:Any = None 
    dataset_test: Any = None 

    if args.dataset == datasetEnum.DatasetType.MNIST.value: 
        dataset_train = datasets.MNIST(dataPath, train = True, transform = transform)
        dataset_test = datasets.MNIST(dataPath, train = False, transform = transform)
        datasetImgDimensions = 28
        numColourChannels = 1 
        numClasses = 10 
    
    dataPath:str = ""
    datasetImgDimensions: int = 0
    numColourChannels: int = 0 
    numClasses: int = 0 

    model = None 

    if args.model_architecture == modelArchEnum.ModelType.CNN.value: 
        model: nn.Module = ConvolutionalNN.ConvolutionalNeuralNetwork(image_dimensions = datasetImgDimensions, input_channels = numColourChannels, output_size = numClasses )
    
    model.to_device(device) 

    optimizer: torch.optim.Optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss()
    scheduler: torch.optim.lr_scheduler._LRScheduler = StepLR(optimizer, step_size = 1, gamma = args.gamma)
    
    #Initial pruning of the dataset 
    

    #__________________________________________________________________________________________
    
    #Corruption of the pruned dataset 
    corrupted_dataset = corruption.corrupt()
    


    # Save the model
    if args.save_model:
        # Define a unique name for the model checkpoint based on the current timestamp
        model_checkpoint_name:str = f"{args.base_folder}/model_params/{model.__class__.__name__}_{timestamp}.pt"
        torch.save(model.state_dict(), model_checkpoint_name)
    




if __name__ == '__main__':
    main()
