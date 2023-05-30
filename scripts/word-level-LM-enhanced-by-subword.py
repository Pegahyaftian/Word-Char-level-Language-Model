import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class CharCNNnetwork(nn.Module):

    def ___init__(self, num_char:int, char_emb_dim:int, cnn_kernels:list):
        
        super().__init__()

        self.char_emb_dim = char_emb_dim

        #char embedding layers
        self.char_embedding = nn.Embedding(num_char, self.char_emb_dim)

        #list of tuples: [(Number of channels, kernel size)]
        self.cnn_kernels = cnn_kernels

        #convolutions of filter with different sizes
        self.convolutions = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                       out_channels = out_channel,
                       kernel_size = (char_emb_dim, kernel_size),
                       bias=True
        ) for out_channel, kernel_size in self.cnn_kernels
        ])

        self.highway_input_dim = sum([x for (x,y) in self.cnn_kernels])
        self.output_tensor_dim = self.highway_input_dim

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim) 




    def conv_layers(self,x):
        chosen_list =[]
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width + 2)
            batch_size = feature_map.shape[0]
            max_feat = torch.max(feature_map, dim=-1)
            # (batch_size, out_channel, 1)
            chosen = max_feat.squeeze()
             # (batch_size, out_channel)
            chosen_list.append(chosen)
        convolved_output = torch.cat(chosen_list,dim= 1)
        # (batch_size, sum(out_channel))

        return convolved_output
    


    def forward(self,x):
        '''
        :param x: Input variable of tensor with shape [batch_size, seq_len, len(word_len + [SOS] +[EOS])]
        : return: Variable of Tensor with shape [batch_size, seq_len, total_num_filters]
        '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = x.reshape(-1, x.shape[-1])
        # (batch_size * seq_len , max_word_len-width + 2)

        x = self.char_embedding(x)
        # (batch_size * seq_len , max_word_len-width + 2, char_emb_dim)

        # x = x.reshape(x.shape[0], 1, x.shape[1], -1)
        x.unsqueeze_(1)
        # (batch_size * seq_len , 1, max_word_len-width + 2, char_emb_dim)

        x = x.transpose(-2, -1)
        # (batch_size * seq_len , 1,  char_emb_dim, max_word_len-width + 2)

        x = self.conv_layers(x)
        # (batch_size * seq_len, sum(out_channel))

        x = self.batch_norm(x)
        # (batch_size * seq_len, sum(out_channel))

        x = self.highway1(x)
        x = self.highway2(x)
        # (batch_size * seq_len, sum(out_channel))

        x = x.reshape(batch_size, seq_len, -1)
        # (batch_size , seq_len, sum(out_channel))

        return x 
        

class Highway(nn.Module):
    '''
    y = t. g(w * x + b) + (1-t) .x
    g is a nonliearity, t is the transform gate [ t = sigmoid(w_t * x + b_t)] and 1-t is the carry gate which allows to directly carry a proportion of the input

    input x of size (batch_size * seq_len, sum(out_channel))
    '''
    def __init__ (self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features = input_size, out_features = input_size, bias = True)
        self.fc2 = nn.Linear(in_features = input_size, out_features = input_size, bias = True)

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        g = F.relu(self.fc2(x))
        y = torch.mul(t,g) + torch.mul(1-t,x)
        return y