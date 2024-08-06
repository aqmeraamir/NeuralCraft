'''
'''
print('file running')

# libraries & modules
import math
import numpy as np
import os
from schematic_manager import read_schematic, create_schematic

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEM_SHAPE = (32, 32, 32)

T = 200
LOAD_MODEL = False

# hyperparameters
EPOCHS = 10000
BATCH_SIZE = 5
LEARNING_RATE = 0.001

# ---------------------------------------
# Configure dataset(s)
# ---------------------------------------

class SchematicsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.schematics = [os.path.join(self.root, schematic_name) for schematic_name in os.listdir(self.root)]
    
    def __len__(self):
        return len(self.schematics)
    
    def __getitem__(self, idx):
        schematic_path = self.schematics[idx]
        blocks, dimensions, _ = read_schematic(schematic_path) 
        blocks_tensor = transform_blocks(blocks, dimensions)

        return blocks_tensor
    
def transform_blocks(blocks, original_dimensions, target_dimensions=SCHEM_SHAPE):
    '''
    transforms the 'blocks' array from a .schematic file to specified dimensions, and turns it into a tensor for pytorch
    '''
    blocks = [-1 if id == 0 else 1 for id in blocks] # normalize values to -1 and 1 respectively
    blocks_tensor = torch.tensor(blocks, dtype=torch.float) # convert to tensor
    blocks_tensor = blocks_tensor.view(original_dimensions) # convert to 3d array
    blocks_tensor = blocks_tensor.unsqueeze(0).unsqueeze(0) # add channel dimension
    blocks_tensor = F.interpolate(blocks_tensor, size=target_dimensions, mode='nearest') # resize 

    return blocks_tensor[0, :, :, :]

def create_schematic_from_tensor(filepath, blocks_tensor, dimensions=SCHEM_SHAPE, data=None):
    # ensure tensor is on cpu
    blocks_tensor = blocks_tensor.cpu().numpy()

    # remove batch and channel dimensions + convert to 1d array
    if len(blocks_tensor.shape) == 5 :
        blocks_tensor = blocks_tensor[0, 0, :, :, :].flatten()
    elif len(blocks_tensor.shape) == 4:
        blocks_tensor = blocks_tensor[0, :, :, :].flatten()

    if data is None: 
        data = data = np.zeros_like(blocks_tensor) # create an empty array of 0s the same size as blocks

    # ensure no invalid values (<0)are in the blocks array
    blocks = np.where(blocks_tensor < 0, 0, np.round(blocks_tensor)).astype(int)
    
    if np.prod(dimensions) != len(blocks):
        print(f"Error creating schematic ({filepath}), invalid dimensions for the given blocks array")
        return -1
    
    create_schematic(filepath, blocks, data, dimensions)


# load the schematics dataset & set the data loader
schematics_dataset = SchematicsDataset(root="src/data/schematics_dataset")
schem_data_loader = DataLoader(schematics_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


#create_schematic_from_tensor("src/data/output_schematics/test.schematic", schematics_dataset[0], SCHEM_SHAPE)


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


# sample noise
# t = torch.Tensor([0]).type(torch.int64)
# noisy_blocks, noise = forward_diffusion_sample(schematics_dataset[1], t)
# create_schematic_from_tensor("src/data/output_schematics/test.schematic", schematics_dataset[0], SCHEM_SHAPE)


# ---------------------------------------
# Backward Process - The 3D U-Net
# ---------------------------------------

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


class Block3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emp_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emp_dim, out_ch)

        if up:
            self.conv1 = nn.Conv3d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose3d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv3d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_ch)
        self.bnorm2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        # convolution 1
        h = self.bnorm1(self.relu(self.conv1(x)))

        # creating and time embedding channel
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None, ) * 3] # add 3 none dimensions
        h += time_emb

        # convolution 2
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        return self.transform(h)
    
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_channels = 1
        time_emb_dim = 32

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_emb_dim), 
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv3d(in_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.downs.append(Block3D(down_channels[i], down_channels[i+1], time_emb_dim))

        self.ups = nn.ModuleList()
        for i in range(len(up_channels)-1):
            self.ups.append(Block3D(up_channels[i], up_channels[i+1], time_emb_dim, up=True))

        self.output = nn.Conv3d(up_channels[-1], out_channels, kernel_size=1)
    
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
    
schem_model = UNet3D()
num_params_schem = sum(p.numel() for p in schem_model.parameters())


# ---------------------------------------
# Training functions
# ---------------------------------------

def save_checkpoint(state, filepath="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filepath)

def load_checkpoint(filepath="src/saved_models/checkpoint.pth.tar"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filepath)
    schem_model.load_state_dict(checkpoint['state_dict'])
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
def sample_schem():
    '''
    Function to create random noise, feed it to the model, and create the output schematic 
    '''

    # Sample noise
    blocks = torch.randn((1, 1, SCHEM_SHAPE[0], SCHEM_SHAPE[1], SCHEM_SHAPE[2]), device=device)

    # compute final schematic with the noise removed
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        blocks = sample_timestep(schem_model, blocks, t)

    # clamp the image to the range [-1, 1] and move it to CPU
    blocks = torch.clamp(blocks, -1.0, 1.0)
    blocks = blocks.detach().cpu()
    create_schematic_from_tensor("src/data/output_schematics/output.schematic", blocks, SCHEM_SHAPE)
    print(f'=> output schematic created')


# ---------------------------------------
# Training - 3D Diffusion Model for MC Schematics
# ---------------------------------------         


# configuring the devices / metal support for mac gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

optimizer = Adam(schem_model.parameters(), LEARNING_RATE)
schem_model.to(device)


# prompt
print("==============================")
print(f"3D minecraft No. Model Params: {num_params_schem} \n\n\
Training on device: {device}")
print("==============================\n")


# load the model, optimizer, and epochs
start_epoch = 0
if LOAD_MODEL:
    start_epoch = load_checkpoint()

# iterate through epochs
for epoch in range(start_epoch, EPOCHS):
    for step, batch in enumerate(schem_data_loader):
        optimizer.zero_grad()
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(schem_model, batch, t)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | step {step:03d} -- Loss: {loss.item():.3f}")
    
        # save the model every x epochs
    if epoch % 50 == 0:
        state = {
            'state_dict': schem_model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        save_checkpoint(state)
        sample_schem()