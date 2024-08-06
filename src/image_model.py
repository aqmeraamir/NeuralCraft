'''
'''

# libraries & modules
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

# constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 64

LOAD_MODEL = False

T = 200

# hyperparameters
EPOCHS = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.001

    
# --------------------------------------
# Configure dataset(s)
# ---------------------------------------

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(self.root, img_name) for img_name in os.listdir(self.root)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: 
            image = self.transform(image)

        return image, 0
    

def show_tensor_image(image):
    '''
    Function to display a transformed, tensor image 
    '''
    # transforms to convert to pillow image from tensor
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
 
    plt.imshow(reverse_transforms(image))


# transforms to resize images to a specific size, rotate it, and turn it into a tensor
image_transforms = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t*2) - 1)
]
image_transform = transforms.Compose(image_transforms)

# test dataset
test_dataset = TestDataset(root="src/data/custom", transform=image_transform) 

# fetch the stanford dataset + combine it into one
stanford_train = torchvision.datasets.StanfordCars(root="src/data/", download=False, transform=image_transform)
stanford_test = torchvision.datasets.StanfordCars(root="src/data/", download=False, transform=image_transform, split='test')
stanford_dataset = torch.utils.data.ConcatDataset([stanford_test, stanford_train])


# set the data loaders
image_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ---------------------------------------
# Forward Process - Noise Schedular
# ---------------------------------------

def cosine_beta_schedule(t, start=0.0001, end=0.02):
    '''
    returns a tensor containing the beta value for each timestep
    '''
    steps = torch.linspace(0, t, t)
    betas = start + 0.5 * (end - start) * (1 - torch.cos(math.pi * steps / t))
    return betas

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="mps"):
    """
    Returns a noisy version of an image at a specific timestep 
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# beta schedule
betas = cosine_beta_schedule(t=T)

# pre-calculate the necessary terms for noise calculations
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# ---------------------------------------
# Backward Process - 2D U-Net 
# ---------------------------------------

class Block2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emp_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emp_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        # convolution 1
        h = self.bnorm1(self.relu(self.conv1(x)))

        # creating and time embedding channel
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None, ) * 2]
        h += time_emb

        # convolution 2
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        return self.transform(h)
    
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_emb_dim), 
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block2D(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block2D(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)
    
    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv0(x)
        
        # U-Net: encoder
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        
        return self.output(x)
    
image_model = UNet()
num_params = sum(p.numel() for p in image_model.parameters())


# ---------------------------------------
# Training functions
# ---------------------------------------

def save_checkpoint(state, filepath="src/saved_models/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filepath)

def load_checkpoint(filepath="src/saved_models/checkpoint.pth.tar"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filepath)
    image_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    return start_epoch

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(model, x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    '''
    Function to create random noise, feed it to the model, and display the output image 
    '''

    # Sample noise
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)

    # compute final image (noise removed)
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(image_model, img, t)

    # clamp the image to the range [-1, 1] and move it to CPU
    img = torch.clamp(img, -1.0, 1.0)
    img = img.detach().cpu()

    # plot the final image
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    show_tensor_image(img)
    plt.show()

# ---------------------------------------=
# Training - 2D Diffusion Model for Images
# ----------------------------------------

# configure the device / metal support for mac gpu
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# prompt
print("==============================")
print(f"2D Image No. Model Params: {num_params} \n\n\
Training on device: {device}")
print("==============================\n")

image_model.to(device)

# configure the optimizer as adam
optimizer = Adam(image_model.parameters(), LEARNING_RATE)

# load the model, optimizer, and epochs
start_epoch = 0
if LOAD_MODEL:
    start_epoch = load_checkpoint()

# iterate through epochs
for epoch in range(start_epoch, EPOCHS):
    for step, batch in enumerate(image_data_loader):
        optimizer.zero_grad()
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(image_model, batch[0], t)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | step {step:03d} -- Loss: {loss.item():.3f}")

    # save + sample an image from the model every x epochs
    if epoch % 200 == 0:
        state = {
            'state_dict': image_model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        save_checkpoint(state)
        sample_plot_image()

