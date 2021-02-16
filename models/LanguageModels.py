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


class RNNTextClassifier(nn.Module):
    
    def __init__(self, text_field, class_field, emb_dim, rnn_size):
        super().__init__()        
        
        voc_size = len(text_field.vocab)
        n_classes = len(class_field.vocab) 
        
        # Embedding layer.
        self.embedding = nn.Embedding(voc_size, emb_dim)

        # If we're using pre-trained embeddings, copy them into the model's embedding layer.
        #if text_field.vocab.vectors is not None:
        #    self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
        #                                               requires_grad=update_pretrained)
        
        # The RNN module: either a basic RNN, LSTM, or a GRU.
        #self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_size, 
        #                  bidirectional=True, num_layers=1)        
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=rnn_size, 
                           bidirectional=True, num_layers=1)
        #self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
        #                  bidirectional=True, num_layers=1)

        # And finally, a linear layer on top of the RNN layer to produce the output.
        self.top_layer = nn.Linear(2*rnn_size, n_classes)
        self.activation = nn.LogSoftmax()
        
    def forward(self, texts):
        # The words in the documents are encoded as integers. The shape of the documents
        # tensor is (max_len, n_docs), where n_docs is the number of documents in this batch,
        # and max_len is the maximal length of a document in the batch.

        # First look up the embeddings for all the words in the documents.
        # The shape is now (max_len, n_docs, emb_dim).
        embedded = self.embedding(texts)
        
        # The RNNs return two tensors: one representing the outputs at all positions
        # of the final layer, and another representing the final states of each layer.
        # In this example, we'll use just the final states.
        # NB: for a bidirectional RNN, the final state corresponds to the *last* token
        # in the forward direction and the *first* token in the backward direction.
        _, (final_state, _) = self.rnn(embedded)
        
        # The shape of final_state is (2*n_layers, n_docs, rnn_size), assuming that 
        # the RNN is bidirectional.
        # We select the top layer's forward and backward states and concatenate them.
        top_forward = final_state[-2]
        top_backward = final_state[-1]
        top_both = torch.cat([top_forward, top_backward], dim=1)
        top_both = self.top_layer(top_both)
        out = self.activation(top_both)
        
        # Apply the linear layer and return the output.
        #print(top_both.shape)
        return out
    
class RNNGate(nn.Module):
    
    def __init__(self, text_field, class_field, emb_dim, rnn_size):
        super().__init__()        
        
        voc_size = len(text_field.vocab)
        n_classes = len(class_field.vocab) 
        
        # Embedding layer.
        self.embedding = nn.Embedding(voc_size, emb_dim)

        # If we're using pre-trained embeddings, copy them into the model's embedding layer.
        #if text_field.vocab.vectors is not None:
        #    self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
        #                                               requires_grad=update_pretrained)
        
        # The RNN module: either a basic RNN, LSTM, or a GRU.
        #self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_size, 
        #                  bidirectional=True, num_layers=1)        
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=rnn_size, 
                           bidirectional=True, num_layers=1)
        #self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
        #                  bidirectional=True, num_layers=1)

        # And finally, a linear layer on top of the RNN layer to produce the output.
        self.top_layer = nn.Linear(2*rnn_size, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, texts):
        # The words in the documents are encoded as integers. The shape of the documents
        # tensor is (max_len, n_docs), where n_docs is the number of documents in this batch,
        # and max_len is the maximal length of a document in the batch.

        # First look up the embeddings for all the words in the documents.
        # The shape is now (max_len, n_docs, emb_dim).
        embedded = self.embedding(texts)
        
        # The RNNs return two tensors: one representing the outputs at all positions
        # of the final layer, and another representing the final states of each layer.
        # In this example, we'll use just the final states.
        # NB: for a bidirectional RNN, the final state corresponds to the *last* token
        # in the forward direction and the *first* token in the backward direction.
        _, (final_state, _) = self.rnn(embedded)
        
        # The shape of final_state is (2*n_layers, n_docs, rnn_size), assuming that 
        # the RNN is bidirectional.
        # We select the top layer's forward and backward states and concatenate them.
        top_forward = final_state[-2]
        top_backward = final_state[-1]
        top_both = torch.cat([top_forward, top_backward], dim=1)
        top_both = self.top_layer(top_both)
        out = self.activation(top_both)
        
        # Apply the linear layer and return the output.
        #print(top_both.shape)
        return out