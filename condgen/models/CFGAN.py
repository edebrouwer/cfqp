import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from condgen.utils import str2bool
import math
import numpy as np
from torchvision.utils import make_grid
import wandb
import pandas as pd
import copy
import plotly.express as px
import pandas as pd
from condgen.models.transformer import TransformerModel

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim, one_D):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.one_D = one_D
    def forward(self, x):
        if self.one_D:
            return self.dense(x)[..., None]
        else:
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

class ImageGenerator(nn.Module):
    def __init__(self,channels = [32, 64, 128, 256]):
        super().__init__()
 

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] , channels[1], 3, stride=2, bias=False, output_padding=1)    
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] , channels[0], 3, stride=2, bias=False, output_padding=1)    
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] , 3, 3, stride=1)
    
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self,x):
        h = self.tconv4(x)
        ## Skip connection from the encoding path
        #h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(h)
        #h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(h)
        #h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(h)
        return h

class ImageEmbedder(nn.Module):
    def __init__(self,channels = [32, 64, 128, 256], in_channels = 1, embed_dim = 128, conditional_len = 28, one_D = False):
        super().__init__()
        # Encoding layers where the resolution decreases
        if one_D:
            self.conv = nn.Conv1d
        else:
            self.conv = nn.Conv2d
        
        out_len = conditional_len
        self.conv1 = self.conv(in_channels, channels[0], 3, stride=1, bias=False)
        out_len = out_len-3 + 1
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = self.conv(channels[0], channels[1], 3, stride=2, bias=False)
        out_len = int((out_len-3)/2 + 1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = self.conv(channels[1], channels[2], 3, stride=2, bias=False)
        out_len = int((out_len-3)/2 + 1)
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = self.conv(channels[2], channels[3], 3, stride=2, bias=False)
        out_len = int((out_len-3)/2 + 1)
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    
        
        if not one_D:
            out_len = out_len * out_len
        out_len *= channels[-1]

        self.out_fun = nn.Sequential(nn.Linear(int(out_len),1024),nn.ReLU(), nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,embed_dim))
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
  
    def forward(self, x): 
        # Encoding path
        h1 = self.conv1(x)    
        ## Incorporate information from t
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        out = h4.view(h4.size(0),-1)
        out = self.out_fun(out)
        return out

   
class NoisePredictor(nn.Module):
    def __init__(self, num_classes = 6, hdim = 32, categorical = True):
        super().__init__()
        """
        categorical : if true, returns a continuous vector of dim num_classes
        """
        embed_dim = 128
        self.image_embedder_x = ImageEmbedder(in_channels = 1, embed_dim = embed_dim)
        self.image_embedder_y = ImageEmbedder(in_channels = 3, embed_dim = embed_dim)
        self.classifier = nn.Sequential(nn.Linear(2*embed_dim + hdim,hdim),nn.ReLU(),nn.Linear(hdim,num_classes))
        self.treatment_embedder = nn.Sequential(nn.Linear(1,hdim),nn.ReLU(),nn.Linear(hdim,hdim))
        self.num_classes = num_classes
        self.categorical = categorical

    def forward(self,X,Y,T, color = None):
        if color is not None:
            classes = torch.nn.functional.one_hot(color[:,0], num_classes = self.num_classes)
            return classes.float()
        else:
            x_embed = self.image_embedder_x(X)
            y_embed = self.image_embedder_y(Y)
            t_embed = self.treatment_embedder(T)
        
            embed = torch.cat((x_embed,y_embed,t_embed),1)
            classes = self.classifier(embed)
            
            if self.categorical:
                return torch.nn.functional.gumbel_softmax(classes, -1)
            else:
                return classes
            #return torch.nn.functional.softmax(classes,-1)


class UnetOutcomePredictor(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], embed_dim=256, one_D = False, out_dims = 1, conditional = False, input_dim = 1):
        """
        if conditional : uses an estimate of the noise variable to predict the outcome
        """
        super().__init__()
            
        if one_D: #Time Series
            self.conv = nn.Conv1d
            self.tconv = nn.ConvTranspose1d
        else:
            self.conv = nn.Conv2d
            self.tconv = nn.ConvTranspose2d

        self.treatment_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        
        if conditional: #EXTRA dimension for the noise estimate
            embed_dim = embed_dim + 1

        # Encoding layers where the resolution decreases
        self.conv1 = self.conv(input_dim, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0], one_D)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = self.conv(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1], one_D)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = self.conv(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2], one_D)
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = self.conv(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3], one_D)
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # Decoding layers where the resolution increases
        self.tconv4 = self.tconv(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2], one_D)
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = self.tconv(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
        self.dense6 = Dense(embed_dim, channels[1], one_D)
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = self.tconv(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
        self.dense7 = Dense(embed_dim, channels[0], one_D)
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = self.tconv(channels[0] + channels[0], out_dims, 3, stride=1)
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
       
        #hdim = int(embed_dim/2)
        #self.treatment_embedder = nn.Sequential(nn.Linear(1,hdim),nn.ReLU(),nn.Linear(hdim,hdim))
        

    def forward(self, X,T, cond = None):
        #x = x.permute(0,2,1)
        embed = self.act(self.treatment_embedder(T[:,0].float()))
        if cond is not None:
            embed = torch.cat((embed,cond),-1)
        # Obtain the Gaussian random feature embedding for t
        #outcomes = []
        #for i_class in torch.arange(self.num_classes,device = X.device):
        #z_embed = self.Embeds(i_class)[None,:].repeat((T.shape[0],1))
        #embed = torch.cat((treat_embed,z_embed),1)
        
        # Encoding path
        h1 = self.conv1(X)    
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
    
        return h


class OutcomePredictor(nn.Module):
    def __init__(self, num_classes = 6):
        super().__init__()
        hdim = 10 #treatment_embedding
        embedding_dim = 8
        embed_img_dim = 128

        self.image_embedder_x = ImageEmbedder(in_channels = 1, embed_dim = embed_img_dim)
        self.image_generator_y = ImageGenerator()
        self.treatment_embedder = nn.Sequential(nn.Linear(1,hdim),nn.ReLU(),nn.Linear(hdim,hdim))

        self.Embeds = nn.Embedding(num_embeddings = num_classes, embedding_dim = embedding_dim)

        self.embed_map = nn.Sequential(nn.Linear(embed_img_dim+hdim+embedding_dim,1024),nn.ReLU(),nn.Linear(1024,1024))
        self.softmax = torch.nn.Softmax()
        self.num_classes = num_classes

    def forward(self,X,T):
        x_embed = self.image_embedder_x(X)
        t_embed = self.treatment_embedder(T)
        embed = torch.cat((x_embed[:,None,:].repeat(1,self.num_classes,1),t_embed[:,None,:].repeat(1,self.num_classes,1),self.Embeds(torch.arange(self.num_classes,device = X.device))[None,...].repeat(X.shape[0],1,1)),2)
        embed_ = self.embed_map(embed)
        embed_ = embed_.view(X.shape[0],self.num_classes,256,2,2)
        imgs = [self.image_generator_y(embed_[:,i]) for i in range(self.num_classes)]
        return torch.stack(imgs,1)


class CFGAN(pl.LightningModule):
    def __init__(
        self,
        #channels,
        num_classes,
        num_classes_model,
        lr,
        model_prev = None,
        continuous_relaxation = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        #data_shape = (channels, width, height)
        #self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        #self.discriminator = Discriminator(img_shape=data_shape)

        if (self.hparams["data_type"] == "SimpleTraj") or (self.hparams["data_type"]=="CV"):
            self.conv1D = True
            out_dims = 2
            input_dim = 2
        else:
            self.conv1D = False
            out_dims = 3
            input_dim = 1

        if num_classes_model == -1:
            self.num_classes = num_classes
        else:
            self.num_classes = num_classes_model
   
        self.continuous_relaxation = continuous_relaxation
        if self.continuous_relaxation:
            self.outcome_predictor = nn.ModuleList([UnetOutcomePredictor(one_D = self.conv1D, out_dims = out_dims, conditional = True, input_dim= input_dim) for _ in range(1)]) #only one model
            self.outcome_predictor_clone = nn.ModuleList([copy.deepcopy(pred) for pred in self.outcome_predictor])
            self.noise_predictor = NoisePredictor(num_classes = 1, categorical = False) # dimension of the noise is 1 (to restrict information)
        else:
           
            if self.hparams["data_type"] == "CV": #using a transformer model for CV
                self.outcome_predictor = nn.ModuleList([TransformerModel(input_long_size = input_dim, hidden_dim = 64, baseline_size = 0,trans_layers = 3, output_dims = out_dims, max_pos_encoding = 100, d_pos_embed = 10, planned_treatments = True, reconstruction_size = out_dims, nheads = 8, ff_dim = 512) for _ in range(self.num_classes)])
                self.outcome_predictor_clone = nn.ModuleList([TransformerModel(input_long_size = input_dim, hidden_dim = 64, baseline_size = 0,trans_layers = 3, output_dims = out_dims, max_pos_encoding = 100, d_pos_embed = 10, planned_treatments = True, reconstruction_size = out_dims, nheads = 8, ff_dim = 512) for _ in range(self.num_classes)])
                self.hard_reset = True
            else:
                self.outcome_predictor = nn.ModuleList([UnetOutcomePredictor(one_D = self.conv1D, out_dims = out_dims, input_dim = input_dim) for _ in range(self.num_classes)])
                self.outcome_predictor_clone = nn.ModuleList([UnetOutcomePredictor(one_D = self.conv1D, out_dims = out_dims, input_dim = input_dim) for _ in range(self.num_classes)])
                self.hard_reset = False
            
            self.noise_predictor = NoisePredictor(num_classes = 1, categorical = True)
        
        #self.discriminator = Discriminator()
        
        self.lr = lr

        self.kmeans = None
        self.second_stage = False #pre-training for initial clustering

        self.update_period = self.hparams["update_period"]

        self.clone()

        self.input_dim = input_dim
        self.out_dims = out_dims

    def clone(self):
        self.outcome_predictor_clone = nn.ModuleList([copy.deepcopy(pred) for pred in self.outcome_predictor])

    def reset_generator(self,num_classes_model, num_classes):

        if num_classes_model == -1:
            self.num_classes = num_classes
        else:
            self.num_classes = num_classes_model
        
        if self.hparams["data_type"] == "CV": #using a transformer model for CV
            self.outcome_predictor = nn.ModuleList([TransformerModel(input_long_size = self.input_dim, hidden_dim = 64, baseline_size = 0,trans_layers = 3, output_dims = self.out_dims, max_pos_encoding = 100, d_pos_embed = 10, planned_treatments = True, reconstruction_size = self.out_dims, nheads = 8, ff_dim = 512) for _ in range(self.num_classes)])
        else:
            self.outcome_predictor = nn.ModuleList([copy.deepcopy(self.outcome_predictor[0]) for _ in range(self.num_classes)])
        
        self.outcome_predictor_clone = nn.ModuleList([copy.deepcopy(pred) for pred in self.outcome_predictor])

    def forward(self, batch):
        img_x, img_y, y, color, T = batch
        T = T[:,None]
        
        #outcome_prediction = self.outcome_predictor(img_x,T)
        if self.continuous_relaxation:
            noise_prediction = self.noise_predictor(img_x,img_y,T, color = self.get_hint(color))
            outcome_prediction = torch.stack([pred(img_x,T, cond = noise_prediction) for pred in self.outcome_predictor],1)
            noise_prediction = torch.ones_like(noise_prediction)
        else:
            outcome_prediction = torch.stack([pred(img_x,T) for pred in self.outcome_predictor],1)
        
            if self.hparams["EM"]:
                clone_predictions = torch.stack([pred(img_x,T) for pred in self.outcome_predictor_clone],1).detach()
                noise_prediction = self.assignment_probs(clone_predictions, img_y)
                
            else:
                noise_prediction = self.noise_predictor(img_x,img_y,T, color = self.get_hint(color))
            
            if self.hparams["cheat_class"]:
                noise_prediction = torch.nn.functional.one_hot(color[:,0])
        
        last_dims = tuple([i for i in range(2,len(img_y.shape)+1)])
        loss_per_class = torch.mean((img_y[:,None,...]-outcome_prediction).pow(2),dim = last_dims)
        loss = (noise_prediction * loss_per_class).sum(1).mean()

        return {"loss":loss, "noise_prediction":noise_prediction, "outcome_prediction":outcome_prediction, "img_x":img_x, "img_y":img_y, "color":color }

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def get_hint(self,color):
        if self.hparams["cheat_class"]:
            hint= color
        else:
            hint = None
        return hint

    def training_step(self, batch, batch_idx):#, optimizer_idx):
        outcome_dict = self(batch) 
        self.log("train_loss", outcome_dict["loss"],on_epoch = True)
        return outcome_dict["loss"]

    def set_kmeans(self,kmeans):
        self.kmeans = kmeans
        self.second_stage = True

    def set_classes(self,num_classes_model):
        self.num_classes = num_classes_model

    def assignment_probs(self,outcome_prediction, img_y):

        if self.kmeans is not None:
            x = (outcome_prediction[:,0] - img_y).reshape(img_y.shape[0],-1)
            labels = self.kmeans.predict(x.detach().cpu())
            assignment_probs = torch.nn.functional.one_hot(torch.LongTensor(labels),num_classes = self.num_classes).to(outcome_prediction.device)
        
        else:
            last_dims = tuple([i for i in range(2,len(img_y.shape)+1)])
            logprobs = -(outcome_prediction.detach()-img_y[:,None,...]).pow(2).mean(dim=last_dims)
        
            #assignment_probs = logprobs / logprobs.sum(1)[:,None]
            assignment_probs = torch.nn.functional.softmax(logprobs,dim = 1)

            assignment_probs = torch.nn.functional.one_hot(assignment_probs.argmax(1),num_classes = self.num_classes)
            #if self.second_stage:
            #    import ipdb; ipdb.set_trace()

        return assignment_probs

    def validation_step(self, batch,batch_idx):
       
        output_dict = self(batch)
        self.log("val_loss", output_dict["loss"],on_epoch = True)
        
        noise_prediction = output_dict["noise_prediction"]
        outcome_prediction = output_dict["outcome_prediction"]
        img_y = output_dict["img_y"]
        img_x = output_dict["img_x"]
        color = output_dict["color"]

        if self.conv1D:
            imgs_out = (noise_prediction[...,None,None] * outcome_prediction).sum(1)
        else:
            imgs_out = (noise_prediction[...,None,None,None] * outcome_prediction).sum(1)
        
        return {"loss":output_dict["loss"], "imgs_pred":imgs_out, "imgs_true":img_y, "imgs_x":img_x, "noise_pred":noise_prediction, "color":color}

    def validation_epoch_end(self,data):
        
        imgs_preds = torch.cat([b["imgs_pred"] for b in data])
        imgs_true = torch.cat([b["imgs_true"] for b in data])
        imgs_x = torch.cat([b["imgs_x"] for b in data]).repeat(1,3,1,1)
        noise_pred = torch.cat([b["noise_pred"] for b in data])
        color = torch.cat([b["color"] for b in data])
        
        noise_idx = noise_pred.argmax(1).cpu()
        
        if not self.conv1D: #print the images
            imgs = torch.cat((imgs_x[:10],imgs_true[:10],imgs_preds[:10]))
            img_array = make_grid(imgs, nrow = 10)
            images = wandb.Image(img_array, caption="IMAGES EXAMPLES")
            self.logger.experiment.log({"examples": images})
        #self.log("categories",hist)

        else: #plot the trajectories
            y = imgs_preds[0,0].detach().cpu()
            df = pd.DataFrame(y,columns  = ["pred"])
            df["time"] = np.arange(len(y))
            df["type"] = "pred"
            df_ = df.copy()
            df_["pred"] =  imgs_true[0,0].detach().cpu()
            df_["type"] = "true"
            df = pd.concat((df,df_))
            fig = px.line(df, x="time", y="pred", title='Predictions', color = "type")
            self.logger.experiment.log({"Predictions Chart": fig})

        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(noise_idx.cpu().numpy(),color.cpu().numpy()[:,0]))
        self.logger.experiment.log({"conf_mat":wandb.plot.confusion_matrix(probs=None,
            y_true=color.cpu().numpy()[:,0], preds=noise_idx.cpu().numpy(),
                        class_names=None)})

        if (((self.current_epoch + 1) % self.update_period == 0) and (self.second_stage)):
            #self.outcome_predictor_clone = nn.ModuleList([copy.deepcopy(pred) for pred in self.outcome_predictor])
            self.clone()
            self.kmeans = None

        return 

    def test_step(self, batch,batch_idx):
        outcome_dict = self(batch) 
        self.log("test_loss", outcome_dict["loss"],on_epoch = True)
        return outcome_dict["loss"]

    def predict_step(self,batch,batch_idx):
        
        if len(batch)==6: #counterfactual evaluation
            x,y_o, color, T_o, T_new, y_new = batch
            y_pred_new, noise_prediction = self.counterfactual_pred(x,y_o,T_o,T_new,color)
            return {"imgs_pred": y_pred_new, "imgs_o_true":y_o, "imgs_x":x, "color":color, "imgs_new_true":y_new, "noise_prediction":noise_prediction,'T_new':T_new,"T_o":T_o}
        
        elif len(batch)==8: #ite evaluation
            x,y_o, color, T_o, T_new, y_new, T_new2, y_new2 = batch
            y_pred_new, noise_prediction = self.counterfactual_pred(x,y_o,T_o,T_new,color)
            y_pred_new2, noise_prediction2 = self.counterfactual_pred(x,y_o,T_o,T_new2,color)
            return {"imgs_pred": y_pred_new, "imgs_o_true":y_o, "imgs_x":x, "color":color, "imgs_new_true":y_new, "noise_prediction":noise_prediction,'T_new':T_new,"T_o":T_o, "T_new2":T_new2,"imgs_pred2":y_pred_new2, "imgs_new2_true":y_new2}
            
        else:
            output_dict = self(batch)
        
            noise_prediction = output_dict["noise_prediction"]
            outcome_prediction = output_dict["outcome_prediction"]
            img_y = output_dict["img_y"]
            img_x = output_dict["img_x"]
            color = output_dict["color"]

            if self.conv1D:
                imgs_out = (noise_prediction[...,None,None] * outcome_prediction).sum(1)
            else:
                imgs_out = (noise_prediction[...,None,None,None] * outcome_prediction).sum(1)
        
            return {"loss":output_dict["loss"], "imgs_pred":imgs_out, "imgs_true":img_y, "imgs_x":img_x, "noise_pred":noise_prediction, "color":color}
            

    def counterfactual_pred(self,img_x,img_y,T, T_new, color):
        
        T = T[:,None]
        T_new = T_new[:,None]

        if self.hparams["EM"]:
            outcome_prediction = torch.stack([pred(img_x,T) for pred in self.outcome_predictor],1)
            noise_prediction = self.assignment_probs(outcome_prediction, img_y)
        else:
            noise_prediction = self.noise_predictor(img_x,img_y,T_new, color = self.get_hint(color))


        if self.hparams["cheat_class"]:
            noise_prediction = torch.nn.functional.one_hot(color[:,0])
        
        outcome_prediction = torch.stack([pred(img_x,T_new) for pred in self.outcome_predictor],1)

        best_noise_prediction = outcome_prediction[np.arange(outcome_prediction.shape[0]),noise_prediction.argmax(1),...]
        return best_noise_prediction, noise_prediction

    def configure_optimizers(self):

        opt_g = torch.optim.Adam(list(self.outcome_predictor.parameters())+list(self.noise_predictor.parameters()), lr=self.lr)
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g], []

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--cheat_class', type=str2bool, default=False)
        parser.add_argument('--EM', type=str2bool, default=False)
        parser.add_argument('--num_classes_model', type=int, default=-1, help = "If -1, will use the same number of classes as for generating the data")
        parser.add_argument('--update_period', type=int, default=10, help = "Number of epochs before updating the class assignments")
        parser.add_argument('--continuous_relaxation', type=str2bool, default=False, help = "If True, infers a continuous prediction of the noise variable and uses it for the treatment prediction")
        return parser
