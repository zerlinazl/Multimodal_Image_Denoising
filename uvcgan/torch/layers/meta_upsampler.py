import torch
from torch import nn

# Small network for metadata

class MetaUpsampler(nn.Module):

    def __init__(
        self, image_shape, **kwargs
    ):

        super().__init__(**kwargs)

        self.norm = nn.BatchNorm2d(1) 
        self.activ = nn.LeakyReLU()
        
        _, self.height, self.width = image_shape

        # make list of layer operations
        self.layers = []

        # upsample from 1D array to array with dims (h, w)
        # future work: modify what's in self.layers to try out different small network structures
        self.layers.append(nn.Upsample(size=(self.height, self.width))) 
        self.layers.append(self.activ)
        self.layers.append(self.norm)

        self.layers = nn.ModuleList(self.layers)

    def cat_reshape(self, x):
        # reshape output to concatenable dims 
        x = torch.reshape(x, (-1, 1, self.height, self.width))
        x = torch.cat([x,x], dim=0)
        return x

    def forward(self, x):
        # expand dimensions to 4D
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)

        # pass through small net
        for layer in self.layers:
            x = layer(x)
            
        # x will be dims: (batch size, num_meta, H_inner, W_inner)
        return self.cat_reshape(x)
            



        
            

