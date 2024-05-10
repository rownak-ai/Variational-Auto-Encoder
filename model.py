import torch
from torch import nn

#Input img -> Hidden dim -> mean, std -> Reparametrization trick -> Decoder ->Output img
class VariationalAutoencoder(nn.Module):
    def __init__(self,input_dimension,hidden_dimension=200,latent_dimension=20):
        super(VariationalAutoencoder,self).__init__()
        #encoder
        self.img_to_hidden = nn.Linear(input_dimension, hidden_dimension)
        self.hidden_to_mean = nn.Linear(hidden_dimension, latent_dimension)
        self.hidden_to_std = nn.Linear(hidden_dimension, latent_dimension)

        #decoder
        self.z_to_hidden = nn.Linear(latent_dimension, hidden_dimension)
        self.hidden_to_img = nn.Linear(hidden_dimension, input_dimension)

        self.relu = nn.ReLU()


    def encoder(self,x):
        #q_phi(z/x)
        h = self.relu(self.img_to_hidden(x))
        mean = self.hidden_to_mean(h)
        std = self.hidden_to_std(h)
        return mean,std

    def decoder(self,z):
        #p_theta(x/z)
        h = self.z_to_hidden(z)
        return torch.sigmoid(self.hidden_to_img(h))

    def forward(self,x):
        mean,std = self.encoder(x)
        epsilon = torch.rand_like(std)
        z_reparameterize = mean + std*epsilon
        x_reconstructed = self.decoder(z_reparameterize)
        return x_reconstructed,mean,std

if __name__ == "__main__":
    x = torch.randn(4,28*28)
    vae = VariationalAutoencoder(input_dimension=784)
    x_reconstructed,mean,std = vae(x)
    print(x_reconstructed.shape)
    print(mean.shape)
    print(std.shape)


