import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      
import matplotlib.pyplot as plt

torch.manual_seed(1)    
class load_data:
    def __init__(self,epoch=1,batch_size=50):
        self.EPOCH = epoch          
        self.BATCH_SIZE = batch_size
    def train(self):
        # Mnist dataset
        train_data = torchvision.datasets.MNIST(
            root='./MNIST_data/',    
            train=True,  # this is training data
            transform=torchvision.transforms.ToTensor(),
            download=True,                                                      
        )        
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)
        return train_loader
    def val(self):
        test_data = torchvision.datasets.MNIST(root='./MNIST_data/', train=False,download=True)        
        test_x = test_data.test_data[:4000].view(-1,1,28,28).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        test_y = test_data.test_labels[:4000]        
        return test_x,test_y
    def test(self):
        test_data = torchvision.datasets.MNIST(root='./MNIST_data/', train=False,download=True)        
        test_x = test_data.test_data[4000:].view(-1,1,28,28).type(torch.FloatTensor)/255.   # shape from (:, 28, 28) to (:, 1, 28, 28), value in range(0,1)
        test_y = test_data.test_labels[4000:]        
        return test_x,test_y


    
