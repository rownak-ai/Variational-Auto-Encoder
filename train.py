import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoencoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 1e-4

dataset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
model = VariationalAutoencoder(INPUT_DIM,H_DIM,Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=LR_RATE)
loss_fun = nn.BCELoss(reduction='sum')

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for batch_idx,(data,_) in loop:
        data = data.to(DEVICE).view(data.shape[0],INPUT_DIM)
        x_reconstructed,mean,std = model(data)
        
        reconstruction_loss = loss_fun(x_reconstructed,data)
        kl_divergence = -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))

        loss = reconstruction_loss + kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


model = model.to('cpu')
def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x,y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break
    encoding_digits = []
    for d in range(10):
        with torch.no_grad():
            mean,std = model.encoder(images[d].view(1,784))
        encoding_digits.append((mean,std))

    mean,std = encoding_digits[digit]

    for example in range(num_examples):
        epsilon = torch.rand_like(std)
        z = mean + std * epsilon
        out = model.decoder(z)
        out = out.view(-1,1,28,28)
        save_image(out,f'C:\\Variational Auto Encoder\\generated_images_{digit}_{example}.png')

for idx in range(10):
    inference(idx,num_examples=10)

