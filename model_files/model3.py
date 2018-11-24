"""
Model3.py : Resnet 152 decoder. Single layer LSTM with 256 dimensional hidden states, 128 dimesnional learnable word embeddings
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
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
    
    def sample_beam(self, features,  end_idx,beam_width = 3, states=None):
        """Generate captions for given image features using beam search."""
        sampled_ids = [[]]*beam_width
        probs = [0]*beam_width
        inputs = features.unsqueeze(1)
        state_list = []
        final_list = []
        
        for i in range(self.max_seg_length):
            if i == 0 :
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1)) 
                prob, predicted = outputs.topk( beam_width)
                prob = prob.cpu().detach().numpy().reshape(-1)
                
                for t in range(beam_width):
                    sampled_ids[t].append( predicted[0][t].unsqueeze(0) )
                    probs[t] += np.log(prob[t] ) 
                    state_list.append(states )
                    
            else :
                temp_outs = []
                for b in range(len(sampled_ids) ):
                    predicted_val = sampled_ids[b][-1]
                    inputs = self.embed(predicted_val).unsqueeze(1)                         
                    states = state_list[b] 
                    
                    hiddens, states = self.lstm(inputs, states )   
                    outputs = self.linear(hiddens.squeeze(1))            
                    prob, predicted = outputs.topk( beam_width)
                    prob = prob.cpu().detach().numpy().reshape(-1)
                
                    for t in range(beam_width):
                        temp_prob = (probs[b] + np.log(prob[t] ) )/ (i+1)
                        temp_list = sampled_ids[b] + [predicted[0][t].unsqueeze(0) ]
                        temp_outs.append([ temp_prob, temp_list, states  ])
                        
                        if predicted[0][t].unsqueeze(0).cpu().detach().numpy()[0] == end_idx:
                            final_list.append([ temp_prob, temp_list, states  ])
                            temp_outs.pop()
                        
                
                temp_outs = sorted(temp_outs, key= lambda x: -x[0])
                sampled_ids = [x[1] for x in temp_outs[:beam_width]]
                probs = [x[0] for x in temp_outs[:beam_width]]
                state_list = [x[2] for x in temp_outs[:beam_width]]
        
        final_list.extend(temp_outs) 
        sampled_ids = sorted(final_list, key=lambda x:-x[0])[0][1]
        sampled_ids = torch.stack(sampled_ids, 1)                
        return sampled_ids