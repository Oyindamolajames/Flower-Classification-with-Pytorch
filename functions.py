import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# Load and process data
def load_process(data_dir):
    '''
    A function to load and process data for image classification.

    Parameters:
    - data_dir (str): The directory containing the train, valid, and test data folders.

    Returns:
    - trainloader, validloader, testloader (DataLoaders): Data loaders for the training, validation, and test sets.
    - train_data, valid_data, test_data (Datasets): ImageFolder datasets for the training, validation, and test sets.
    
    This function loads image data from the specified directory and applies appropriate transformations
    for training, validation, and testing sets. It uses the ImageFolder dataset from torchvision.

    Transforms:
    - Training data: Randomly rotated, resized, and cropped to 224x224 pixels. Random horizontal flip applied.
    - Validation and test data: Resized and center-cropped to 224x224 pixels.

    Normalization:
    - All sets are normalized using the ImageNet mean and standard deviation.

    Data Loaders:
    - Training data loader shuffles the data and uses a batch size of 64.

    Example Usage:
    ```python
    trainloader, validloader, testloader, train_data, valid_data, test_data = load_process('path/to/data')
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    # transform for train data
    train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # transform for validation data
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # transform for test data
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    # load the train data 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    # load the validation data
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # load the test data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 
    validloader= torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    print("Data loaded and processed successfully...")
    return trainloader, validloader, testloader, train_data, valid_data, test_data

# Flatten operation
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)    

# Define a function to add classifier to pretrained model
def add_classifier(model, input_units, hidden_units, dropout):
    """
    Modify the classifier of a pre-trained PyTorch model by replacing it with a custom classifier.

    Parameters:
    - model (nn.Module): A pre-trained PyTorch model.
    - input_units (int): The number of input units for the custom classifier.
    - hidden_units (int): The number of hidden units in the custom classifier.
    - dropout (float): The dropout probability for regularization in the custom classifier.

    Returns:
    - nn.Module: The modified model with the custom classifier.

    This function takes a pre-trained PyTorch model and replaces its classifier with a custom
    classifier tailored for a specific task. It freezes the parameters of the original model
    to retain pre-trained features and only trains the newly added custom classifier.

    The custom classifier consists of a linear layer with ReLU activation, dropout, and a final
    linear layer with LogSoftmax activation for multi-class classification.

    Example Usage:
    ```python
    import torchvision.models as models

    # Load a pre-trained ResNet model
    pretrained_model = models.resnet18(pretrained=True)

    # Add a custom classifier with 256 input units, 128 hidden units, and 0.5 dropout
    model_with_classifier = add_classifier(pretrained_model, 256, 128, 0.5)
    """
    print("Training model in progress...")
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        


    model.classifier = nn.Sequential(Flatten(),
                                     nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    return model

# define function to train the model
def train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu_mode, device):
    '''
    Train a PyTorch model on a given dataset.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be trained.
    - epochs (int): The number of training epochs.
    - trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - validloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - criterion (torch.nn.Module): Loss function for training.
    - optimizer (torch.optim.Optimizer): Optimization algorithm for updating model parameters.
    - gpu_mode (bool): If True, the model will be trained on GPU.
    - device (torch.device): The device (CPU or GPU) on which to train the model.

    Returns:
    - torch.nn.Module: The trained model.
    - torch.optim.Optimizer: The updated optimizer state.

    This function takes a PyTorch model, along with training and validation dataloaders,
    and performs training for the specified number of epochs. It uses the provided loss
    criterion and optimizer to update the model's parameters. Optionally, it can train
    the model on GPU if the `gpu_mode` is set to True.

    Example Usage:
    ```python
    import torch.optim as optim
    from torchvision import models

    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Set up optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model for 5 epochs
    trained_model, updated_optimizer = train_model(model, 5, trainloader, validloader, criterion, optimizer, gpu_mode=True, device=torch.device('cuda'))
    ```

    Note:
    - Ensure that the model, criterion, and optimizer are initialized before calling this function.
    - `gpu_mode` specifies whether to use GPU for training. If True, the model will be moved to the GPU.
    - `device` specifies the device on which to perform training (torch.device('cuda') for GPU or torch.device('cpu') for CPU).
    '''
    model.to(device)
    
    if gpu_mode == True:
        model.to('cuda')

    # TODO: Build and train your network
    # epochs = 3
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        logps = model(inputs)

                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    #calculate accuracy
                    ps = torch.exp(logps)
                    top_class = ps.argmax(dim=1)
                    equals = top_class == labels
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. | Train loss: {running_loss/len(trainloader):.3f}.. | Valid loss:            {valid_loss/len(validloader):.3f}.. | Test accuracy: {accuracy/len(validloader) * 100:.3f}"
                     )
    return model, optimizer  

# define the test model function
def test_model(model, testloader, gpu_mode, device):
    '''
    Evaluate the performance of a PyTorch model on a test dataset.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - gpu_mode (bool): If True, the model will be evaluated on GPU.
    - device (torch.device): The device (CPU or GPU) on which to evaluate the model.

    This function takes a PyTorch model and a DataLoader for the test dataset, and evaluates
    the model's accuracy on the test set. It optionally moves the model to GPU if `gpu_mode` is True.

    Example Usage:
    ```python
    from torchvision import models

    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Test the model on the test set
    test_model(model, testloader, gpu_mode=True, device=torch.device('cuda'))
    ```

    Note:
    - Ensure that the model has been trained or loaded with appropriate weights before testing.
    - `gpu_mode` specifies whether to use GPU for evaluation. If True, the model will be moved to the GPU.
    - `device` specifies the device on which to perform evaluation (torch.device('cuda') for GPU or torch.device('cpu') for CPU).
    '''
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        model.to('cpu')
        
    # TODO: Do validation on the test set
    correct_img = 0
    total_img = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)

            _, predicted = torch.max(logps.data, 1)
            total_img += labels.size(0)
            correct_img += (predicted == labels).sum().item()


    print(f"Accuracy is {correct_img/total_img * 100}%")
    

# define function that saves model
def save_model(model, train_data, architecture, input_units, learning_rate, epochs,  optimizer,  save_dir):
    '''
    Save a PyTorch model checkpoint along with relevant training details.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be saved.
    - architecture (str): The architecture of the model.
    - input_units (int): The number of input units in the model's classifier.
    - learning_rate (float): The learning rate used during training.
    - epochs (int): The number of training epochs.
    - optimizer (torch.optim.Optimizer): The optimizer used during training.

    This function takes a PyTorch model, its architecture, training details, and optimizer, and
    saves a checkpoint file containing essential information for later use, such as model state,
    classifier architecture, learning rate, and more.

    Example Usage:
    ```python
    from torchvision import models
    import torch.optim as optim

    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Save the model checkpoint
    save_checkpoint(model, 'resnet18', 512, 0.001, 5, optimizer)
    ```

    Note:
    - Ensure that the model, architecture, input_units, learning_rate, epochs, and optimizer
      are initialized before calling this function.
    - The saved checkpoint file ('checkpoint.pth') can be used for model loading or
      transfer learning in the future.
    '''
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'archictecture': architecture,
                  'input_size': input_units,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                 }

    torch.save(checkpoint, 'checkpoint.pth')

    print("Checkpoint saved succesfully..")
    

# define a function to load trained model
def load_checkpoint(path):
    '''
    Load a PyTorch model checkpoint from a specified file.

    Parameters:
    - path (str): The path to the checkpoint file.

    Returns:
    - torch.nn.Module: The PyTorch model loaded from the checkpoint.

    This function loads a PyTorch model checkpoint from a specified file and returns the model.
    The checkpoint file should contain essential information such as model architecture, classifier,
    optimizer state, model state_dict, class_to_idx mapping, and other relevant details.

    Example Usage:
    ```python
    # Load a previously saved checkpoint file
    loaded_model = load_checkpoint('checkpoint.pth')
    ```
    '''
    
    checkpoint = torch.load(path)
    

    
    model = getattr(models, checkpoint['archictecture'])(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# define function the preprocess the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Load the image with PIL
    image = Image.open(image)
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # apply transformation on the image
    image_trns = transform(image)
    
    # convert to numpy array
    image_np = np.array(image_trns)
    
    # transpose the colour channel to be first dimension
    image_np = image_np.transpose((0, 2, 1))
    
    # convert the numpy array to a pytorch tensor
    torch_image = torch.from_numpy(image_np)
    
    return torch_image

# define function that predict 
def predict(image_path, model, topk=5, gpu_mode):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    print("Predicting image...\n")
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        model.to('cpu')
    
    # get processed img 
    img = process_image(image_path).unsqueeze(0)
    
    
    model.eval()
    with torch.no_grad():
        logps = model.forward(img)
        
    # calculate probabilities
    
    ps = torch.exp(logps)
    
    probs, labels = ps.topk(topk)
    
    probs = ps.topk(topk)[0]
    index = ps.topk(topk)[1]
    
    top_prob = np.array(probs[0])
    top_labels = np.array(index[0])
    
    class_to_idx = model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}
    
    top_labels_list = []
    for i in top_labels:
        top_labels_list.append(indx_to_class[i])
    
    
    return top_prob, top_labels_list

