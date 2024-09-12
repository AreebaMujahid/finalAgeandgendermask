import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

class UTKDataset(Dataset):
    '''
        Inputs:
            dataFrame : Pandas dataFrame
            transform : The transform to apply to the dataset
    '''
    def __init__(self, dataFrame, transform=None):
        # read in the transforms
        self.transform = transform
        
        # Use the dataFrame to get the pixel values
        data_holder = dataFrame.pixels.apply(lambda x: np.array(x.split(" "),dtype=float))
        arr = np.stack(data_holder)
        arr = arr / 255.0
        arr = arr.astype('float32')
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        # reshape into 48x48x1
        self.data = arr
        
        # get the age, gender, and ethnicity label arrays
        self.age_label = np.array(dataFrame.bins[:])        # Note : Changed dataFrame.age to dataFrame.bins with most recent change
        self.gender_label = np.array(dataFrame.gender[:])
        self.eth_label = np.array(dataFrame.ethnicity[:])
    
    # override the length function
    def __len__(self):
        return len(self.data)
    
    # override the getitem function
    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index]
        data = self.transform(data)
        
        # load the labels into a list and convert to tensors
        labels = torch.tensor((self.age_label[index], self.gender_label[index], self.eth_label[index]))
        
        # return data labels
        return data, labels
    

# High level feature extractor network (Adopted VGG type structure)
class highLevelNN(nn.Module):
    def __init__(self):
        super(highLevelNN, self).__init__()
        self.CNN = nn.Sequential(
            # first batch (32)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # second batch (64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Third Batch (128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.CNN(x)

        return out

# Low level feature extraction module
class lowLevelNN(nn.Module):
    def __init__(self, num_out):
        super(lowLevelNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=num_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, padding=1))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class TridentNN(nn.Module):
    def __init__(self, num_age, num_gen, num_eth):
        super(TridentNN, self).__init__()
        # Construct the high level neural network
        self.CNN = highLevelNN()
        # Construct the low level neural networks
        self.ageNN = lowLevelNN(num_out=num_age)
        self.genNN = lowLevelNN(num_out=num_gen)
        self.ethNN = lowLevelNN(num_out=num_eth)

    def forward(self, x):
        x = self.CNN(x)
        age = self.ageNN(x)
        gen = self.genNN(x)
        eth = self.ethNN(x)

        return age, gen, eth
    
'''
    Function to test the trained model

    Inputs:
      - testloader : PyTorch DataLoader containing the test dataset
      - modle : Trained NeuralNetwork
    
    Outputs:
      - Prints out test accuracy for gender and ethnicity and loss for age
'''
def test(testloader, model):
  device = 'cuda' if torch.cuda.is_available() else 'cpu' 
  size = len(testloader.dataset)
  # put the moel in evaluation mode so we aren't storing anything in the graph
  model.eval()

  age_acc, gen_acc, eth_acc = 0, 0, 0

  with torch.no_grad():
      for X, y in testloader:
          X = X.to(device)
          age, gen, eth = y[:,0].to(device), y[:,1].to(device), y[:,2].to(device)
          pred = model(X)

          age_acc += (pred[0].argmax(1) == age).type(torch.float).sum().item()
          gen_acc += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
          eth_acc += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

  age_acc /= size
  gen_acc /= size
  eth_acc /= size

  print(f"Age Accuracy : {age_acc*100}%,     Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n")


import numpy as np

# Read in the dataframe
dataFrame = pd.read_csv(r'C:\Users\Laiba Ahmad\Downloads\age_gender.gz', compression='gzip')
 
# Construct age bins
age_bins = [0,10,15,20,25,30,40,50,60,120]
age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
dataFrame['bins'] = pd.cut(dataFrame.age, bins=age_bins, labels=age_labels)

# Split into training and testing
train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)

# get the number of unique classes for each group
class_nums = {'age_num':len(dataFrame['bins'].unique()), 'eth_num':len(dataFrame['ethnicity'].unique()),
              'gen_num':len(dataFrame['gender'].unique())}

# Define train and test transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49,), (0.23,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49,), (0.23,))
])

# Construct the custom pytorch datasets
train_set = UTKDataset(train_dataFrame, transform=train_transform)
test_set = UTKDataset(test_dataFrame, transform=test_transform)

# Load the datasets into dataloaders
trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
testloader = DataLoader(test_set, batch_size=128, shuffle=False)

# Sanity Check
for X, y in trainloader:
    print(f'Shape of training X: {X.shape}')
    print(f'Shape of y: {y.shape}')
    break 


# Configure the device 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)

# Define the list of hyperparameters
hyperparameters = {'learning_rate':0.001, 'epochs':30}

# Initialize the TridentNN model and put on device
model = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
model.to(device)

'''
  Functions to load and save a PyTorch model
'''
def save_checkpoint(state, epoch):
  print("Saving Checkpoint")
  filename = "tridentNN_epoch"+str(epoch)+".pth.tar"
  torch.save(state,filename)

def load_checkpoint(checkpoint):
  print("Loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  opt.load_state_dict(checkpoint['optimizer'])


'''
train the model
''' 
# Load hyperparameters
learning_rate = hyperparameters['learning_rate']
num_epoch = hyperparameters['epochs']

# Define loss functions
age_loss = nn.CrossEntropyLoss()
gen_loss = nn.CrossEntropyLoss() # TODO : Explore using Binary Cross Entropy Loss?
eth_loss = nn.CrossEntropyLoss()

# Define optimizer
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epoch):
  # Construct tqdm loop to keep track of training
  loop = tqdm(enumerate(trainloader), total=len(trainloader), position=0, leave=True)
  age_correct, gen_correct, eth_correct, total = 0,0,0,0

  # save the model every 10 epochs
  if epoch % 10 == 0:
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : opt.state_dict(), 
                  'age_loss' : age_loss, 'gen_loss' : gen_loss, 'eth_loss' : eth_loss}
    save_checkpoint(checkpoint, epoch)

  # Loop through dataLoader
  for _, (X,y) in loop:
    # Unpack y to get true age, eth, and gen values
    # Have to do some special changes to age label to make it compatible with NN output and Loss function
    #age, gen, eth = y[:,0].resize_(len(y[:,0]),1).float().to(device), y[:,1].to(device), y[:,2].to(device)
    age, gen, eth = y[:,0].to(device), y[:,1].to(device), y[:,2].to(device)
    X = X.to(device)
    pred = model(X)          # Forward pass
    loss = age_loss(pred[0],age) + gen_loss(pred[1],gen) + eth_loss(pred[2],eth)   # Loss calculation

    # Backpropagation
    opt.zero_grad()          # Zero the gradient
    loss.backward()          # Calculate updates

    # Gradient Descent
    opt.step()               # Apply updates

    # Update num correct and total
    age_correct += (pred[0].argmax(1) == age).type(torch.float).sum().item()
    gen_correct += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
    eth_correct += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

    total += len(y)

    # Update progress bar
    loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
    loop.set_postfix(loss = loss.item())

  # Update epoch accuracy
  gen_acc, eth_acc, age_acc = gen_correct/total, eth_correct/total, age_correct/total

  # print out accuracy and loss for epoch
  print(f'Epoch : {epoch+1}/{num_epoch},    Age Accuracy : {age_acc*100},    Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n')


test(testloader, model)

