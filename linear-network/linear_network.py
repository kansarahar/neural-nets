import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

################################# args ################################

parser = argparse.ArgumentParser(description='A linear model for the MNIST dataset')

# dataset params
parser.add_argument('--download', dest='download', action='store_true', help='OPTIONAL - use to download the mnist dataset (default: false)')
parser.add_argument('--dataset_location', dest='dataset_location', type=str, default='../datasets', help='OPTIONAL - the relative path from this folder to the dataset folder (default: ../datasets)')

# training params
parser.add_argument('--train', dest='train', action='store_true', help='OPTIONAL - use if training, otherwise testing (default: false)')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='OPTIONAL - batch size for training (default: 1)')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='OPTIONAL - learning rate (default: 0.001)')
parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='OPTIONAL - number of epochs (default: 1)')

# model params
parser.add_argument('--model_name', dest='model_name', type=str, default='model.pt', help='OPTIONAL - the name of the model to save/load (default: model.pt)')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='OPTIONAL - overwrites an existing model of the same name during training (default: false)')

args = parser.parse_args()

############################## file paths ##############################

dir_name = os.path.dirname(os.path.abspath(__file__))

# directory where models will be stored
os.makedirs(os.path.join(dir_name, 'saved_models'), exist_ok=True)
model_path = os.path.join(dir_name, 'saved_models', args.model_name)

################################ model ################################

class linear_model(nn.Module):
  def __init__(self):
    super(linear_model, self).__init__()
    self.fc_layer1 = nn.Sequential(
      nn.Linear(1*1*28*28, 64),
      nn.ReLU(),
    )
    self.output_layer = nn.Sequential(
      nn.Linear(64, 10),
      nn.LogSoftmax(1),
    )

  def forward(self, x):
    out = x.reshape(x.size(0), -1)
    out = self.fc_layer1(out)
    out = self.output_layer(out)
    return out

loss_function = nn.NLLLoss()
model = linear_model()

########################### transformations ###########################

transformations = transforms.Compose([
    transforms.ToTensor()
])

############################## training ###############################

if args.train:
  dataset_location = os.path.join(dir_name, args.dataset_location)
  train_dataset = torchvision.datasets.MNIST(root=dataset_location, train=True, download=args.download, transform=transformations)

  print('Training model', args.model_name)
  try:
    if (not args.overwrite):
      model.load_state_dict(torch.load(model_path))
      print('Model loaded successfully')
    else:
      print('Overwriting existing model', args.model_name)
  except:
    print('No model like that exists - creating new model')
  
  optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
  for epoch in range(1, args.epochs+1):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0
    for iteration, (image,label) in enumerate(train_loader):
      optimizer.zero_grad()
      loss = loss_function(model(image), label.long())
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      num_iterations=1000
      if (iteration%num_iterations == 0 and iteration > 0):
        torch.save(model.state_dict(), model_path)
        print('Saved model to', os.path.relpath(model_path), '- avg loss:', total_loss/num_iterations)
        total_loss = 0
    print('Epoch', epoch, 'completed')
  sys.exit(1)

############################### testing ###############################

if not args.train:
  dataset_location = os.path.join(dir_name, args.dataset_location)
  test_dataset = torchvision.datasets.MNIST(root=dataset_location, train=False, download=args.download, transform=transformations)
  print('Testing Model', args.model_name)
  try:
    model.load_state_dict(torch.load(model_path))
    print('Model loaded successfully')
  except:
    sys.exit('No model like that exists')
  
  model.eval()
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
  
  correct = 0
  total = 0
  for image, label in test_loader:
      total += 1
      pred = torch.argmax(model(image)).item()
      if pred == label.item():
          correct += 1
  accuracy = (correct*1.0)/total
  print('final accuracy: ', accuracy)
  sys.exit(1)