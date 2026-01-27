from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_dim = 32 * 32 * 3, num_classes = 10):
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, num_classes, bias= False)  
        self.fc2 = nn.Linear(num_classes, num_classes, bias= False)

    def forward(self, x):
        x = self.flatten(x)  
        x = self.fc1(x) 
        x = self.fc2(x)  
        return x