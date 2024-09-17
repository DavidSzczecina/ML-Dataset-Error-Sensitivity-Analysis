import os
import torch
import argparse
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
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

'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
'''

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
            if row[order] in ['test', 'train', 'validation']:
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


def save_mislabeled_images(dataset, label_issues, hdf5_path, output_dir, pred_probabilities):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for rank, index in enumerate(label_issues, start=1):
        image, label, image_name = dataset[index]
        image = getImgData(image_name)
        
        # Decode the label to get the actual word
        label_word = dataset.label_encoder.inverse_transform([label])[0]

        # Get predicted probability for this image
        predicted_prob = pred_probabilities[index]
        predicted_class = np.argmax(predicted_prob)
        certainty = predicted_prob[predicted_class] * 100  # Get the certainty percentage
        
        # Plot and save the image with the label, accuracy, and filename in the title
        plt.imshow(image)
        plt.title(f'Rank: {rank}, Order: {label_word}, Certainty: {certainty:.1f}%, Filename: {image_name}')
        plt.axis('off')

        # Save the image to the specified directory
        save_path = os.path.join(output_dir, f"{rank}_{image_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()  # Clear the figure for the next image

def evaluate_model(model, testloader, loss_function, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    fold_probabilities = []
    fold_labels = []

    with torch.no_grad():
        for inputs, targets, _ in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            fold_probabilities.append(probabilities.cpu().numpy())
            fold_labels.append(targets.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = 100.0 * correct / total

    # Concatenate all predictions and labels
    fold_pred_probabilities = np.concatenate(fold_probabilities, axis=0)
    fold_true_labels = np.concatenate(fold_labels, axis=0)
    
    return avg_loss, accuracy, fold_pred_probabilities, fold_true_labels


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
  parser.add_argument("--loss_function", default="focal_loss", help="loss function to use, default 'focal_loss")
  
  args = parser.parse_args()
  
  # Configuration options
  k_folds = args.num_folds
  num_epochs = args.epochs
  weights = args.weights #resnet50
  
  
  '''
  alpha_values = [ # normalized and apporximated based on distribution of BIOSCAN-1M
    0.037, 0.037, 0.054, 0.037, 0.003, 0.037, 
    0.037, 0.059, 0.029, 0.085, 0.259, 0.259, 
    0.037, 0.259, 0.259
]

  
  if args.loss_function == 'focal_loss':
    loss_function = FocalLoss(alpha=alpha_values, gamma=2.0)
  else:
    loss_function = nn.CrossEntropyLoss()
  '''
  loss_function = nn.CrossEntropyLoss()
  
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

  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
  
  # Start print
  print('--------------------------------')
  startTime = time.time()
  
  #Concatted Fold results
  results = {}
  total_pred_probabilities = []
  total_y_pred = []
  total_y_true = []
  
  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset), 1):
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

    #Push data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        # Move the model to the GPU
        model.to(device)
    
    
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
      
      # Set current loss value
      current_loss = 0.0
      model.train()  # Set the model to training mode
      
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
        
        # Update Loss
        current_loss += loss.item()
      
      # Print Statistics
      avg_train_loss = current_loss / num_batches
      
      # Evaluate the model on the test data
      test_loss, test_accuracy, fold_pred_probabilities, fold_true_labels = evaluate_model(model, testloader, loss_function, device)
      print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')
    
    # Print fold results
    print(f'FOLD {fold}')
    print('--------------------------------')
    print(f'Accuracy for fold {fold}: {test_accuracy:.2f}%')
    print(f'Loss for Fold: {test_loss:.3f}')
    print('--------------------------------')
    results[fold] = test_accuracy

    #Per fold details
    class_names = dataset.label_encoder.classes_
    y_true = fold_true_labels
    y_pred = np.argmax(fold_pred_probabilities, axis=1)
    
    #add per fold values to complete list
    total_pred_probabilities.extend(fold_pred_probabilities)
    total_y_pred.extend(y_pred)
    total_y_true.extend(fold_true_labels)
    
    '''
    print("Unique classes in y_true:", set(y_true))
    print("Unique classes in y_pred:", set(y_pred))
    print("Number of classes:", len(set(y_true)))
    print("Number of target names:", len(class_names))
    '''

    class_names = dataset.label_encoder.classes_
    labels = list(range(len(class_names)))  
    
    # Calculate and print classification report
    report = classification_report(y_true, y_pred, target_names=class_names, labels=labels, zero_division=0.0)
    print('Classification Report:')
    print(report)
    
    '''
    # Concat the predicetd fold_probabilities
    pred_probabilities.extend(np.concatenate(fold_probabilities, axis=0))
    labels.extend(np.concatenate(y_true, axis=0))
    '''
    
    
    # Saving the model
    if args.save_model == True:     
      path = args.save_model_path 
      save_path = f'{path}-fold-{fold}.pth'
      print('Saving trained model :', save_path)
      torch.save(model.state_dict(), save_path)
      
      
  
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  print("Duration was :", time.time()-startTime)
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  avg_accuracy = np.mean(list(results.values()))
  print(f'Average Accuracy: {avg_accuracy:.2f}%')
  
  

  pruning_filter = "both"
  labels = total_y_true
  pred_probabilities = np.array(total_pred_probabilities)
  label_issues = find_label_issues(
      labels,
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
    save_mislabeled_images(dataset, label_issues, hdf5_path, output_dir, pred_probabilities)

