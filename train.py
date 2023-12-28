import argparse
import torch
from torchvision import models
from torch import nn
from torch import optim

from functions import load_process, add_classifier, train_model, test_model, save_model

parser = argparse.ArgumentParser(description="Argument parser for the model" )

# add argument that contains data directory
parser.add_argument('--data_dir', type=str, default='flowers', 
                    help='Path to the data directory. Default is flowers.')

# add arguement for model to be pretrained
parser.add_argument('--arch', action='store', dest='pretrained_model', default='densenet121',
                   help='Enter the naem of the pretrained model, which can either be densenet or vgg archictectures.\
                         the default archictecture name is densenet121')

# add argument for hidden inputs
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=500,
                    help='Enter the value for hidden units. The default is 500.')

# add argument for dropout rate
parser.add_argument('--dropout', action='store', dest='dropout', type=float, default=0.5,
                   help='Enter the dropout rate. The default is 0.5' )

# add argument for learning rate
parser.add_argument('--learnin_rate', action='store', dest='lr', type=float, default=0.003,
                   help='Enter the learning rate. The default is 0.001')

# add argument for epochs
parser.add_argument('--epochs', action='store', dest='ep', type=int, default=4, 
                   help='Enter the epochs. The default is 3')

# add argument to on gpu
parser.add_argument('--gpu', action='store', dest='gpu', default=False,
                   help='Enter the gpu mode true or false. The default mode is False')

# add argument for save dir
parser.add_argument('--save_dir', action='store', dest='save_dir', default='checkpoint.pth',
                   help='Enter the location to save the model. Default is checkpoint.pth')

inputs = parser.parse_args()

data_dir = inputs.data_dir

# load and process data
trainloader, validloader, testloader, train_data, valid_data, test_data = load_process(data_dir)

# use GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download a pretrained model
model = getattr(models, inputs.pretrained_model)(pretrained=True)

# add classifier to the pretrained model
input_units = model.classifier.in_features
model = add_classifier(model, input_units, inputs.hidden_units, inputs.dropout)

# define loss function
criterion = nn.NLLLoss()

# train classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=inputs.lr)

# train the model
model, optimizer = train_model(model, inputs.ep, trainloader, validloader, criterion, optimizer, inputs.gpu, device)

# test model
test_model(model, testloader, inputs.gpu, device)

# save model
save_model(model, train_data, inputs.pretrained_model, input_units, inputs.lr, inputs.ep,  optimizer,  inputs.save_dir)
