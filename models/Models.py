import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x
    

class GateSigmoid(nn.Module):
    def __init__(self, dim_in):
        super(GateSigmoid, self).__init__()
        self.layer_input = nn.Linear(dim_in, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.activation(x)
        return x
    
class MLP2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP2, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x

class GateMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(GateMLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out2
    
class GateCNN(nn.Module):
    def __init__(self, args):
        super(GateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
class GateCNNSoftmax(nn.Module):
    def __init__(self, args):
        super(GateCNNSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
class CNNFashion(nn.Module):
    def __init__(self, args):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, args.num_classes)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out2

    
class GateCNNFashion(nn.Module):
    def __init__(self, args):
        super(GateCNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
class GateCNNFahsionSoftmax(nn.Module):
    def __init__(self, args):
        super(GateCNNSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 3)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class LSTMClassifier(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super(LSTMClassifier, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
        #self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        #self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 8)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8, 4)
        self.activation = nn.LogSoftmax()
        
    def forward(self, x, offsets):
        x = self.embeddings(x,offsets)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x.view(len(x), 1, -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.activation(x)
        return out
    
class LSTMGate(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super(LSTMGate, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
        #self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        #self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 8)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8,1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x, offsets):
        x = self.embeddings(x,offsets)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x.view(len(x), 1, -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.activation(x)
        return out