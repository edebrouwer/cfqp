import torch
import numpy as np
import torch.nn as nn
from condgen.models.CFGAN import ImageEmbedder

class Dense1D(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None]

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ConditionalEmbedder(nn.Module):
    def __init__(self,embed_dim, in_channels, one_D, conditional_len ):
        """
        in_channels  : number of channels in the image
        """
        super().__init__()
        self.treatment_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        self.image_embedder = ImageEmbedder(in_channels = in_channels,embed_dim = embed_dim, one_D = one_D, conditional_len = conditional_len)
        self.act = lambda x: x * torch.sigmoid(x)

        self.output_weight = nn.Linear(2*embed_dim, embed_dim)

    def forward(self,X,T):
        # X is an image and T is a scalar
        treat_embed = self.act(self.treatment_embedder(T.float()))
        image_embed = self.image_embedder(X)

        return self.output_weight(torch.cat((treat_embed,image_embed),-1))

class ConditionalScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, input_dim = 1, output_dim = 3, one_D = False, conditional_len = 28):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()

    
    self.conditional_embed = ConditionalEmbedder(embed_dim = int(embed_dim/2), in_channels = input_dim, one_D = one_D, conditional_len = conditional_len)
    self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, int(embed_dim/2)))

    if one_D: #Time Series
        self.conv = nn.Conv1d
        self.tconv = nn.ConvTranspose1d
        self.dense = Dense1D
    else:
        self.conv = nn.Conv2d
        self.tconv = nn.ConvTranspose2d
        self.dense = Dense

    # Encoding layers where the resolution decreases
    self.conv1 = self.conv(output_dim, channels[0], 3, stride=1, bias=False)
    self.dense1 = self.dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = self.conv(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = self.dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = self.conv(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = self.dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = self.conv(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = self.dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = self.tconv(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = self.dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = self.tconv(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = self.dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = self.tconv(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = self.dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = self.tconv(channels[0] + channels[0], output_dim, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t, X_cond, T_cond): 
    # Obtain the Gaussian random feature embedding for t 
    embed = self.conditional_embed(X_cond, T_cond)

    embed = torch.cat((self.act(self.time_embed(t)),embed),-1)
    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[(...,) + (None,) * (len(h.shape)-1) ]
    return h


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, conditional_score = False, conditional_dim = 0):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()

    
    self.conditional_score = conditional_score
    if conditional_score:
        self.conditional_embed = nn.Sequential(nn.Linear(conditional_dim,embed_dim), nn.ReLU(), nn.Linear(embed_dim,int(embed_dim/2)))
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, int(embed_dim/2)))
    else:
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))


    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t, cond = None): 
    # Obtain the Gaussian random feature embedding for t 
    if self.conditional_score:
        embed = torch.cat((self.act(self.embed(t)),self.conditional_embed(cond)),-1)
    else:
        embed = self.act(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h


class CNNEmbedder(nn.Module):
    def __init__(self, channels = [32,64,128], conditional_dim  = None, conditional_len = None, embed_dim = None, static_dim = None ):
        super().__init__()
        # Encoding layers where the resolution decreases
        out_len = conditional_len
        self.conv1 = nn.Conv1d(conditional_dim, channels[0], 3, stride=1, bias=False)
        out_len = out_len-2-1 + 1
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], 3, stride=2, bias=False)
        out_len = (out_len-2-1)/2 + 1
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], 3, stride=2, bias=False)
        out_len = (out_len-2-1)/2 + 1
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.act = lambda x: x * torch.sigmoid(x)

        self.static_fun = nn.Sequential(nn.Linear(static_dim,64),nn.ReLU(), nn.Linear(64,32))
        self.out_fun = nn.Linear(channels[-1]*int(out_len) + 32,embed_dim)
    
    def forward(self,x):# Encoding path
        (b,x) = x
        h1 = self.conv1(x)    
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h3_ = h3.view(h3.shape[0],-1)
        embed_static = self.static_fun(b)
        out = self.out_fun(torch.cat((h3_,embed_static),-1))

        return out

class TemporalScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, conditional_score = False, conditional_dim = 0, conditional_len = 0, output_dim = 0, static_dim = 0):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    
    self.conditional_score = conditional_score
    self.output_dim = output_dim
    if conditional_score:
        self.conditional_embed = CNNEmbedder(conditional_dim = conditional_dim, conditional_len = conditional_len, embed_dim = embed_dim, static_dim = static_dim)
        self.treatment_embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
        #nn.Sequential(nn.Linear(conditional_dim,embed_dim), nn.ReLU(), nn.Linear(embed_dim,int(embed_dim/2)))
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
        
        embed_dim = 3*embed_dim
    else:
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))


    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv1d(output_dim, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense1D(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv1d(channels[0], channels[1], 3, stride=1, bias=False)
    self.dense2 = Dense1D(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv1d(channels[1], channels[2], 3, stride=1, bias=False)
    self.dense3 = Dense1D(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv1d(channels[2], channels[3], 3, stride=1, bias=False)
    self.dense4 = Dense1D(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], 3, stride=1, bias=False)
    self.dense5 = Dense1D(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose1d(channels[2] + channels[2], channels[1], 3, stride=1, bias=False, output_padding=0)    
    self.dense6 = Dense1D(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose1d(channels[1] + channels[1], channels[0], 3, stride=1, bias=False, output_padding=0)    
    self.dense7 = Dense1D(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose1d(channels[0] + channels[0], self.output_dim, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t, cond = None, T_treat = None):
    #x = x.permute(0,2,1)
    #if cond is not None:
    #    cond = cond.permute(0,2,1)
    # Obtain the Gaussian random feature embedding for t

    if self.conditional_score:
        treat_embed = self.treatment_embed(T_treat)
        embed = torch.cat((self.act(self.embed(t)),self.conditional_embed(cond), treat_embed),-1)
    else:
        embed = self.act(self.embed(t))   
    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None]
    return h

