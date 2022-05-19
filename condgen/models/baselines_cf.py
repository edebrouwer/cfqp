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
from condgen.models.CFGAN import UnetOutcomePredictor



class CFBaseline(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        num_classes_model,
        lr,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        #data_shape = (channels, width, height)
        #self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        #self.discriminator = Discriminator(img_shape=data_shape)

        if self.hparams["data_type"] == "SimpleTraj":
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
  
        self.outcome_predictor = UnetOutcomePredictor(one_D = self.conv1D, out_dims = out_dims, input_dim = input_dim)

        
        self.lr = lr


    def forward(self, batch):
        img_x, img_y, y, color, T = batch
        T = T[:,None]
        
        #outcome_prediction = self.outcome_predictor(img_x,T)
        
        outcome_prediction = self.outcome_predictor(img_x,T)
        loss = (img_y - outcome_prediction).pow(2).mean()

        return {"loss":loss, "outcome_prediction":outcome_prediction, "img_x":img_x, "img_y":img_y, "color":color }


    def training_step(self, batch, batch_idx):#, optimizer_idx):
        outcome_dict = self(batch) 
        self.log("train_loss", outcome_dict["loss"],on_epoch = True)
        return outcome_dict["loss"]
    
    def validation_step(self, batch,batch_idx):
       
        output_dict = self(batch)
        self.log("val_loss", output_dict["loss"],on_epoch = True)
        
        outcome_prediction = output_dict["outcome_prediction"]
        img_y = output_dict["img_y"]
        img_x = output_dict["img_x"]
        color = output_dict["color"]
        
        imgs_out = outcome_prediction
        
        return {"loss":output_dict["loss"], "imgs_pred":imgs_out, "imgs_true":img_y, "imgs_x":img_x, "color":color}

    def validation_epoch_end(self,data):
        
        imgs_preds = torch.cat([b["imgs_pred"] for b in data])
        imgs_true = torch.cat([b["imgs_true"] for b in data])
        imgs_x = torch.cat([b["imgs_x"] for b in data]).repeat(1,3,1,1)
        color = torch.cat([b["color"] for b in data])
        
        
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

        return 

    def test_step(self, batch,batch_idx):
        outcome_dict = self(batch) 
        self.log("test_loss", outcome_dict["loss"],on_epoch = True)
        return outcome_dict["loss"]

    def predict_step(self,batch,batch_idx):
        
        if len(batch)==6: #counterfactual evaluation
            x,y_o, color, T_o, T_new, y_new = batch
            y_pred_new, _ = self.counterfactual_pred(x,y_o,T_o,T_new)
            return {"imgs_pred": y_pred_new, "imgs_o_true":y_o, "imgs_x":x, "color":color, "imgs_new_true":y_new}
            
        else:
            img_x, img_y, y, color, T = batch
        
            T = T[:,None]
        
            outcome_prediction = self.outcome_predictor(img_x,T)
            noise_prediction = torch.Tensor([0])
        
            return {"imgs_pred": outcome_prediction, "imgs_true":img_y, "imgs_x":img_x, "color":color, "noise_prediction":noise_prediction}


    def counterfactual_pred(self,img_x,img_y,T, T_new):
        
        T = T[:,None]
        T_new = T_new[:,None]

        outcome_prediction = self.outcome_predictor(img_x,T_new) 

        return outcome_prediction, None

    def configure_optimizers(self):

        opt_g = torch.optim.Adam(list(self.outcome_predictor.parameters()), lr=self.lr)
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g], []

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--num_classes_model', type=int, default=-1, help = "If -1, will use the same number of classes as for generating the data")
        return parser
