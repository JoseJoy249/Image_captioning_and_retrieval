"""
Model2.py : Resnet 101 encoder, single Layer LSTM with 512 hidden states, 256 dimensional, learnable word embeddings
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    def sample_beam(self, features, beam_width = 3, states=None):
        """Generate captions for given image features using beam search."""
        sampled_ids = [[]]*beam_width
        probs = [0]*beam_width
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            if i == 0 :
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1)) 
                for t in range(beam_width):
                    prob, predicted = outputs.max(t+1)
                    sampled_ids.append(predicted)
                    probs.append( np.log(prob) )
            else :
                temp_outs = []
                for b in range(beam_width):
                    predicted = sampled_ids[b][-1]
                    inputs = self.embed(predicted)                       
                    inputs = inputs.unsqueeze(1)  

                    hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
                    outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)

                    for t in range(beam_width):
                        prob, predicted = outputs.max(t+1)
                        new_prob = probs[b] + np.log(prob)
                        new_id = sampled_ids[b].append(predicted)
                        temp_outs.append([new_prob, new_id])

                temp_outs = temp_outs.sorted(key= lambda x:x[0])[:beam_width]
                sampled_ids = [x[1] for x in temp_outs]
                probs = [x[0] for x in temp_outs]
        
        sampled_ids = sorted(zip(probs, sampled_ids), key = lambda x:x[0] )[0][1]
        
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids