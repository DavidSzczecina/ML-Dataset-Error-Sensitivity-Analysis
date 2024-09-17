import os
import torch
import argparse
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import warnings
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
import time 
from cleanlab.filter import find_label_issues
import h5py
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import io
from torch.cuda.amp import GradScaler, autocast


# Set a seed for reproducibility
def setSeed(seed):
    np.random.seed(seed)  # using the same seed for numpy and torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    warnings.filterwarnings(
        "ignore", "Lazy modules are a new feature.*"
    )  # ignore warning related to lazy modules

def reset_weights():
  '''
    Try resetting model weights to avoid weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, hdf5_path, tsv_path, order, order_split, transform=None):
        self.hdf5_path = hdf5_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transform
        
        # Parse the TSV file to filter out the correct split data
        self.data = []
        df = pd.read_csv(tsv_path, sep='\t')
        for index, row in df.iterrows():
          if order_split == 'all':
            if row[order] != 'no_split':
              self.data.append((row['order'], row['image_file']))
          elif row[order] == order_split:
            if row['image_file'] != "BIOUG65943-E04.jpg": #apparently this image is not in the HDF5 file
              self.data.append((row['order'], row['image_file']))

        self.label_encoder = LabelEncoder()
        # Extract labels from the validation data and encode them
        self.labels = self.label_encoder.fit_transform([x[0] for x in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the label and image file name for the current index
        label = self.labels[idx]
        image_name = self.data[idx][1]

        # Load the image from the HDF5 file
        image = self.getImg(image_name)
        
        # Convert the image to a tensor and apply transformations if needed
        #image = torch.tensor(image, dtype=torch.float32).to(self.device)

        if self.transform:
            image = self.transform(image)

        # If images are single-channel, add a channel dimension
        if len(image.shape) == 2:
            image = torch.unsqueeze(image, 0)

        return image, label, image_name

    def getImg(self, image_name):
        with h5py.File(self.hdf5_path, 'r') as hdf5:
            group_name = "bioscan_dataset"
            if group_name in hdf5.keys():
                hdf5 = hdf5[group_name]
            image = np.array(hdf5[image_name])
            image_data = Image.open(io.BytesIO(image))
            img_array = np.array(image_data)
            return img_array
          
    

def getImgData(image_name):
    hdf5 = h5py.File(hdf5_path, 'r')
    group_name = "bioscan_dataset"
    if group_name in hdf5.keys():
        hdf5 = hdf5[group_name]
    image = np.array(hdf5[image_name])
    image_data = Image.open(io.BytesIO(image))
    img_array = np.array(image_data)
    return img_array



def save_mislabeled_images(dataset, label_issues, hdf5_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for rank, index in enumerate(label_issues, start=1):
        image, label, image_name = dataset[index]
        image = getImgData(image_name)
        
        # Decode the label to get the actual word
        label_word = dataset.label_encoder.inverse_transform([label])[0]

        # Plot and save the image with the label in the title
        plt.imshow(image)
        plt.title(f'Rank: {rank}, Order: {label_word}, Filename: {image_name}')
        plt.axis('off')

        # Save the image to the specified directory
        save_path = os.path.join(output_dir, f"{rank}_{image_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()  # Clear the figure for the next image


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Bioscan Cleaning Code")
  parser.add_argument("--base_model", default="resnet50", help="base model to train, Default: resnet50")
  parser.add_argument("--seed", type=int, default = 321, help="for consistency while testing")
  parser.add_argument("--epochs", type=int, default = 1, help="Number of epochs to run for")
  parser.add_argument("--num_folds", type=int, default = 5, help="Number of k folds to compute")
  parser.add_argument("--save_model", dest='save_model', default=False, action='store_true', help="For Saving the current Model")
  parser.add_argument("--weights", default='resnet50', help="weights to use for training model")
  parser.add_argument("--update_model", dest='update_model', default=False, action='store_true',  help="Whether to continue training previous model")
  parser.add_argument("--save_model_path", default="testModel", help="path to model weights if continuing to train model")
  parser.add_argument("--num_classes", type=int, default = 16, help="number of classes in dataset")
  parser.add_argument("--batch_size", type=int, default = 64, help="batch size for data loader")
  parser.add_argument("--order", default = 'small_insect_order', help="order to use as data, ie small_insect_order, medium... or large...")
  parser.add_argument("--order_split", default = 'all', help="order split to use, ie test, train validation or all")
  parser.add_argument("--don't_predicted_labels", dest='no_predicted_labels', default=False, action='store_true',  help="Whether to not save identified erroneous labels")
  
  
  
  args = parser.parse_args()
  
  # Configuration options
  k_folds = args.num_folds
  num_epochs = args.epochs
  loss_function = nn.CrossEntropyLoss()
  weights = args.weights #resnet50
  
  
  transform = transforms.Compose([
    transforms.Lambda(lambda image: Image.fromarray(image)),  # Convert NumPy array to PIL Image
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
  ])

  
  # Set fixed random number seed
  setSeed(args.seed)
  hdf5_path = 'cropped_256.hdf5'
  tsvPath = 'BIOSCAN_Insect_Dataset_metadata.tsv'
  print("Loading data")
  dataset_data = np.load('dataset_data.npy')
  dataset_labels = np.load('dataset_labels.npy')
  data_path = "dataset_data.npy"
  labels_path = "dataset_labels.npy"
  order = args.order
  order_split = args.order_split
  dataset = CustomDataset(hdf5_path, tsvPath, order, order_split, transform=transform)
    
  '''
  # Prepare MNIST dataset by concatenating Train/Test part; we split later.
  dataset_train_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
  dataset_test_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
  dataset = ConcatDataset([dataset_train_part, dataset_test_part])
  '''
  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    
  # Start print
  print('--------------------------------')
  startTime = time.time()
  
  # For fold results
  results = {}
  pred_probabilities = []
  labels = []
  
  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    foldStartTime = time.time()
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=args.batch_size,
                      num_workers = 8,
                      sampler=train_subsampler,
                      pin_memory=True,
                      prefetch_factor=6)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=args.batch_size, 
                      num_workers = 8,
                      sampler=test_subsampler,
                      pin_memory=True,
                      prefetch_factor=4)
    
    # Init the neural network    
    model = models.resnet50()
    # Load the saved weights
    if args.update_model == True:
      path = args.save_model_path
      save_path = f'{path}-fold-{fold}.pth'
      print(save_path)
      num_classes = args.num_classes  # Change this to the number of classes in your dataset
      model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
      model.load_state_dict(torch.load(save_path))
    elif weights == 'resnet50':
      model.load_state_dict(torch.load('resnet50_weights.pth'))
      num_classes = args.num_classes  # Change this to the number of classes in your dataset
      model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #criteria = CrossEntropyLoss()
    if torch.cuda.is_available():
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        #criteria.cuda()
        # Move the model to the GPU
        model.to(device)
        # Move the dataset to the GPU if needed
        #dataset_data_GPU = torch.tensor(dataset_data_formatted).to(device)
        #dataset_labels_GPU = torch.tensor(dataset_labels_encoded).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
      epochStartTime = time.time()
      num_batches = len(trainloader)
      
      total_images = len(trainloader.dataset)
      if epoch == 0:
        print(f"Training on {total_images} images") 
      # Print epoch
      print(f'Starting epoch {epoch+1}')
      
      # Set current loss value
      current_loss = 0.0
      
        
      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        
        # Get inputs and targets, and move them to the device
        inputs, targets, image_name = data
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()        
        
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Print statistics
        current_loss += loss.item()
        if i == num_batches-1:
            print('Epoch time :',time.time()-epochStartTime,' Loss in epoch: %.3f' %
                  (current_loss / num_batches))
            current_loss = 0.0
    
    # Saving the model
    if args.save_model == True:     
      path = args.save_model_path 
      save_path = f'{path}-fold-{fold}.pth'
      print('Saving trained model :', save_path)
      torch.save(model.state_dict(), save_path)

    fold_probabilities = []
    fold_labels = []

    # Evaluation for this fold
    correct, total = 0, 0
    fold_loss = 0.0
    with torch.no_grad():
      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):
        # Get inputs and targets, and move them to the device
        inputs, targets, label_name = data
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate outputs
        outputs = model(inputs)
        
        # Calculate Loss
        loss = loss_function(outputs, targets)
        fold_loss += loss.item()
        
        # Get predicted probabilities
        probabilities = torch.softmax(outputs, dim=1)
        fold_probabilities.append(probabilities.cpu().numpy())
        fold_labels.append(targets.cpu().numpy())
        
        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
      # Concat the predicetd fold_probabilities
      pred_probabilities.extend(np.concatenate(fold_probabilities, axis=0))
      labels.extend(np.concatenate(fold_labels, axis=0))
      # Print accuracy
      # Average fold loss
      fold_loss /= len(testloader)
      print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
      print('Loss for Fold: ', fold_loss)
      print('Fold duration: ', time.time()-foldStartTime)
      print('--------------------------------')
      results[fold] = 100.0 * (correct / total)
  
  pred_probabilities = np.array(pred_probabilities)
  
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  print("Duration was :", time.time()-startTime)
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  print(f'Average: {sum/len(results.items())} %')
  
  
  #print(pred_probabilities)
  """
    pruning filter options:
    'prune_by_class', 
    'prune_by_noise_rate',
    'both',
    'confident_learning', 
    'predicted_neq_given', 
    'low_normalized_margin', 
    'low_self_confidence'   
    Default:  'prune_by_noise_rate')
    """
  pruning_filter = "both"

  label_issues = find_label_issues(
      labels,#change to properly formatted labels as these might not be in order
      pred_probabilities,
      return_indices_ranked_by="self_confidence",
      filter_by=pruning_filter,  # change filter here
      frac_noise=1,
  )
  print(f"Label issues: {label_issues}")
  print("Amount of label issues :",len(label_issues))
  
  if args.no_predicted_labels == False:
    matplotlib.use('Agg')
    path = args.save_model_path
    output_dir = 'mislabeledData_' + path
    print("Saving in: ", output_dir)
    save_mislabeled_images(dataset, label_issues, hdf5_path, output_dir)

