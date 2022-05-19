

class ScoreMatcher(pl.LightningModule):
    def __init__(self,weight_decay, lr, sigma, conditional_score, conditional_dim, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.sigma = sigma

        self.conditional = conditional_score

        self.marginal_prob_std_fn = functools.partial(self.marginal_prob_std, sigma=self.sigma)
    
        self.embed_x = CNNEmbedder(conditional_dim = 1, conditional_len)
        self.pred_model = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn, conditional_score = conditional_score, conditional_dim = conditional_dim)
        self.predict_x = 0
        self.noise_inference = 0
        
        self.diffusion_coeff_fn = functools.partial(self.diffusion_coeff, sigma=self.sigma)
        self.score_model = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn, conditional_score = conditional_score, conditional_dim = conditional_dim)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    
    def marginal_prob_std(self,t, sigma):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  

        Returns:
        The standard deviation.
        """    
        #t = torch.tensor(t, device=device)
        return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

    def diffusion_coeff(self,t, sigma):
        """Compute the diffusion coefficient of our SDE.

        Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

        Returns:
        The vector of diffusion coefficients.
        """
        return sigma**t

    def loss_fn(self,x,y,eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
        model: A PyTorch model instance that represents a 
          time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
          the perturbation kernel.
        eps: A tolerance value for numerical stability.
        """

        #infer the color
        if optimizer_idx == 0: 
            col_hat = self.noise_inference(x,y, treat)
            regul_loss = - self.adversarial_loss(self.predict_x(col_hat, treat))
            pred = self.pred_model(x,treat,col_hat)
            pred_loss = self.prediction_loss(pred,y)
            loss = regul_loss + pred_loss
        else:
            pred_x = self.predict_x(col_hat, treat)
            regul_loss = self.adversarial_loss(self.predict_x(col_hat, treat))
            loss = regul_loss


        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t, self.sigma)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.score_model(perturbed_x, random_t, cond = y)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss


    def process_condition(self,y):
        return y 

    def validation_step(self,batch,batch_idx):
        x,y,treat = batch
        loss = self.loss_fn(x, y, treat)
        self.log("val_loss", loss,on_epoch = True)
        return loss

    def training_step(self,batch,batch_idx):
        x,y = batch
        loss = self.loss_fn(x, self.process_condition(y))
        self.log("train_loss", loss,on_epoch = True)
        return loss

    def test_step(self,batch,batch_idx):
        x,y = batch
        loss = self.loss_fn(x, self.process_condition(y))
        self.log("test_loss", loss,on_epoch = True)
        return loss

    def sample(self, sampler, sample_batch_size, cond = None):
        samples = sampler(self.score_model, self.marginal_prob_std_fn, self.diffusion_coeff_fn, sample_batch_size , cond = cond)
        return samples

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--sigma', type=float, default=25.0)
        parser.add_argument('--conditional_score', type=strtobool, default=False)
        return parser

