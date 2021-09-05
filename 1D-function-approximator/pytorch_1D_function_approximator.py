import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

################################# args ################################

dir_name = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='A 1D function approximator playground built with pytorch')

# training params
parser.add_argument('--train', dest='train', action='store_true', help='OPTIONAL - use if training, otherwise testing (default: false)')
parser.add_argument('--lr', dest='lr', type=float, default=0.05, help='OPTIONAL - learning rate (default: 0.005)')
parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='OPTIONAL - dropout probability (default: 0)')
parser.add_argument('--epochs', dest='epochs', type=int, default=3, help='OPTIONAL - number of epochs (default: 3)')

# model params
parser.add_argument('--hidden_nodes', dest='hidden_nodes', type=int, default=4, help='OPTIONAL - number of nodes in hidden layer (default: 4)')
parser.add_argument('--model_name', dest='model_name', type=str, default='model', help='OPTIONAL - the name of the model to save/load (default: model)')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='OPTIONAL - overwrites an existing model of the same name during training (default: false)')

args = parser.parse_args()

################################ model ################################

class FunctionApproximator(nn.Module):
  def __init__(self):
    super(FunctionApproximator, self).__init__()
    self.hidden_layer = nn.Sequential(
      nn.Linear(1, args.hidden_nodes),
      nn.Sigmoid(),
      nn.Dropout(p=args.dropout)
    )
    self.output_layer = nn.Sequential(
      nn.Linear(args.hidden_nodes, 1),
    )

  def forward(self, x):
    out = self.hidden_layer(x)
    out = self.output_layer(out)
    return out

model = FunctionApproximator()
loss_function = nn.MSELoss()

# directory where models will be stored
os.makedirs('%s/models/' % os.path.abspath(dir_name), exist_ok=True)
model_path = os.path.join(dir_name, 'models', args.model_name)

############################## function ###############################

# define function to approximate
def g(x):
  return np.sin(4*x)**2

# define the range of values over which to approximate g
x_values = np.linspace(0, 1, 10000)

shuffled_x_values = x_values.copy()
np.random.shuffle(shuffled_x_values)
data = [(torch.tensor([x]).float(), torch.tensor([g(x)]).float()) for x in shuffled_x_values]

############################## training ###############################

if args.train:
  print('Training Model', args.model_name)
  try:
    if (not args.overwrite):
      model.load_state_dict(torch.load(model_path))
      print('Model loaded successfully')
  except:
    print('No model with that name exists - creating new model')
  
  optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
  for epoch in range(args.epochs):
    total_loss = 0

    for x,y in data:
      optimizer.zero_grad()
      loss = loss_function(model(x), y)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    print('Saving Model to ', args.model_name, ' ...')
    torch.save(model.state_dict(), model_path)
    print('Model Saved!')
    print('epoch', epoch, 'completed, total loss:', total_loss)

############################### testing ###############################

if not args.train:
  print('Testing Model', args.model_name)
  try:
    model.load_state_dict(torch.load(model_path))
    print('Model loaded successfully')
  except:
    sys.exit('No model with that name exists')
  
  model.eval()
  total_loss = 0
  approximation = np.zeros(x_values.shape)
  with torch.no_grad():
    for i in range(len(x_values)):
      x = torch.tensor([x_values[i]]).float()
      approximation[i] = model(x).item()
  
  plt.plot(x_values, g(x_values))
  plt.plot(x_values, approximation)

  # directory where images will be stored
  os.makedirs('%s/images/' % os.path.abspath(dir_name), exist_ok=True)
  plt.savefig('%s/images/%s' % (dir_name, 'pytorch_1D_approximation.png'))
  print('Image saved')

