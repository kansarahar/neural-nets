import os
import numpy as np
import matplotlib.pyplot as plt

dir_name = os.path.dirname(os.path.abspath(__file__))

######################### activation functions ########################

class ReLU:
  def out(self, arr):
    return np.array([arr[i] if arr[i] > 0 else 0 for i in range(len(arr))])
  def derivative(self, arr):
    return np.array([1 if arr[i] > 0 else 0 for i in range(len(arr))])

class Logistic:
  def out(self, arr):
    return np.array([1/(1+np.exp(-1*arr[i])) for i in range(len(arr))])
  def derivative(self, arr):
    f = self.out(arr)
    return np.array([f[i]*(1-f[i]) for i in range(len(arr))])


################################ model ################################

class Model:
  def __init__(self, hidden_nodes, af):
    
    self.w = np.random.rand(hidden_nodes)
    self.b1 = np.random.rand(hidden_nodes)
    self.activation_function = af
    self.c = np.random.rand(hidden_nodes)
    self.b2 = np.random.rand()

  def forward(self, x):
    Z = self.w*x + self.b1
    H = self.activation_function.out(Z)
    return self.c.T@H+self.b2
  
  def step(self, x, y, lr):
    Z = self.w*x + self.b1
    H = self.activation_function.out(Z)
    f = self.c.T@H+self.b2
    loss = (f-y)**2

    df = 2*(f-y)
    db2 = df
    dc = df*H
    dH = df*self.c
    db1 = dH*self.activation_function.derivative(Z)
    dw = db1*x

    self.w -= lr*dw
    self.b1 -= lr*db1
    self.c -= lr*dc
    self.b2 -= lr*db2

    return loss

############################### function ##############################

def g(x):
  return np.sin(x)**2

# range of x values [a, b]
def generate_point_within_range(a, b):
  return np.random.rand()*(b-a)+a

########################### initialize model ##########################

relu = ReLU()
logistic = Logistic()
model = Model(4, logistic)

################################ train ################################

data_points = 10000
lr_base = 0.1
x_range = [0,3]
cumulative_loss = 0
for i in range(data_points):
  x = generate_point_within_range(x_range[0], x_range[1])
  lr = lr_base * (data_points-i)/data_points

  loss = model.step(x, g(x), lr)
  cumulative_loss += loss
  if (i%1000 == 0):
    print(cumulative_loss)
    cumulative_loss = 0

################################# plot ################################

x = np.linspace(x_range[0],x_range[1],100)
y = g(x)
f = np.array([model.forward(point) for point in x])

plt.plot(x, y)
plt.plot(x, f)

os.makedirs('%s/images/' % os.path.abspath(dir_name), exist_ok=True)
plt.savefig('%s/images/%s' % (dir_name, '1D_approximation.png'))
print('Image saved')