import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from netvlad import NetVLAD, NetRVLAD


class Model(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Model, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size

        print('Window size frame:', self.window_size_frame)
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(self.input_size, 512)
            input_size = 512
            self.input_size = 512

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(self.window_size_frame, stride=1)
            self.fc = nn.Linear(input_size, self.num_classes+1)

        if self.pool == "MAX++":
            self.pool_layer_before = nn.MaxPool1d(int(self.window_size_frame/2), stride=1)
            self.pool_layer_after = nn.MaxPool1d(int(self.window_size_frame/2), stride=1)
            self.fc = nn.Linear(2*input_size, self.num_classes+1)


        if self.pool == "AVG":
            self.pool_layer = nn.AvgPool1d(self.window_size_frame, stride=1)
            self.fc = nn.Linear(input_size, self.num_classes+1)

        if self.pool == "AVG++":
            self.pool_layer_before = nn.AvgPool1d(int(self.window_size_frame/2), stride=1)
            self.pool_layer_after = nn.AvgPool1d(int(self.window_size_frame/2), stride=1)
            self.fc = nn.Linear(2*input_size, self.num_classes+1)


        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(cluster_size=self.vlad_k, feature_size=self.input_size,
                                            add_batch_norm=True)
            self.fc = nn.Linear(input_size*self.vlad_k, self.num_classes+1)

        elif self.pool == "NetVLAD++":
            self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.fc = nn.Linear(input_size*self.vlad_k, self.num_classes+1)



        elif self.pool == "NetRVLAD":
            self.pool_layer = NetRVLAD(cluster_size=self.vlad_k, feature_size=self.input_size,
                                            add_batch_norm=True)
            self.fc = nn.Linear(input_size*self.vlad_k, self.num_classes+1)

        elif self.pool == "NetRVLAD++":
            self.pool_layer_before = NetRVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.pool_layer_after = NetRVLAD(cluster_size=int(self.vlad_k/2), feature_size=self.input_size,
                                            add_batch_norm=True)
            self.fc = nn.Linear(input_size*self.vlad_k, self.num_classes+1)


        elif self.pool == 'ATTENTION':
            self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=8, batch_first=True)
            self.pool_layer = nn.AvgPool1d(self.window_size_frame, stride=1)
            self.fc = nn.Linear(input_size, self.num_classes+1)

        elif self.pool == 'ATTENTION_2':
            # Attention Layer
            # Linear
            # Pooling
            # Attention Layer  
            # Pooling Layer 
            self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=8,
             batch_first=True)
            self.linear = nn.Linear(input_size, input_size)
            self.pool_layer = nn.AvgPool1d(self.window_size_frame, stride=1)
            self.attention_2 = nn.MultiheadAttention(embed_dim=input_size, num_heads=8,
             batch_first=True)
            self.fc = nn.Linear(input_size, self.num_classes+1)
            

        self.drop = nn.Dropout(p=0.4)
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):
        # input_shape: (batch,frames,dim_features)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = self.feature_extractor(inputs)
            inputs = inputs.reshape(BS, FR, -1)

        # Aqui acontece a distinção do pooling
        # Temporal pooling operation
        if self.pool == "MAX" or self.pool == "AVG":
            inputs_pooled = self.pool_layer(inputs.permute((0, 2, 1))).squeeze(-1)

        elif self.pool == "MAX++" or self.pool == "AVG++":
            nb_frames_50 = int(inputs.shape[1]/2)    
            input_before = inputs[:, :nb_frames_50, :]        
            input_after = inputs[:, nb_frames_50:, :]  
            inputs_before_pooled = self.pool_layer_before(input_before.permute((0, 2, 1))).squeeze(-1)
            inputs_after_pooled = self.pool_layer_after(input_after.permute((0, 2, 1))).squeeze(-1)
            inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)


        elif self.pool == "NetVLAD" or self.pool == "NetRVLAD":
            inputs_pooled = self.pool_layer(inputs)

        elif self.pool == "NetVLAD++" or self.pool == "NetRVLAD++":
            nb_frames_50 = int(inputs.shape[1]/2)
            inputs_before_pooled = self.pool_layer_before(inputs[:, :nb_frames_50, :])
            inputs_after_pooled = self.pool_layer_after(inputs[:, nb_frames_50:, :])
            inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)

        # faria sentido implementar um attention++?
        # acredito que a camada de atenção ja tenha
        # um aspecto temporal embutido
        elif self.pool == "ATTENTION":
            # The pytorch class returns the output states (same shape as input)
            #  and the weights used in the attention process.
            # print(inputs.shape)
            # selfattention - key, query and value are the same
            inputs_pooled, _ = self.attention(inputs, inputs, inputs)
            inputs_pooled = self.pool_layer(inputs.permute((0, 2, 1))).squeeze(-1)
            # print(type(inputs_pooled))
            # print(inputs_pooled.shape)

        elif self.pool == "ATTENTION_2":
            inputs_pooled, _ = self.attention(inputs, inputs, inputs)
            inputs_pooled = self.linear(inputs_pooled)
            inputs_pooled = self.pool_layer(inputs.permute((0, 2, 1))).squeeze(-1)
            inputs_pooled, _ = self.attention_2(inputs_pooled, inputs_pooled, inputs_pooled)

        # Extra FC layer with dropout and sigmoid activation
        # print('Input: ', inputs_pooled.shape) 
        output = self.sigm(self.fc(self.drop(inputs_pooled)))
        # print('Output: ', output.shape)

        return output


if __name__ == "__main__":
    BS =256
    T = 15
    framerate= 2
    D = 512
    pool = "NetRVLAD++"
    model = Model(pool=pool, input_size=D, framerate=framerate, window_size=T)
    print(model)
    inp = torch.rand([BS,T*framerate,D])
    print(inp.shape)
    output = model(inp)
    print(output.shape)