'''
'''

# libraries and modules
import os
import logging

from tqdm import tqdm
from schematic_manager import read_schematic, create_schematic
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEM_SHAPE = (32, 32, 32)
RUN_NAME = "entire_dataset"
TRAIN = True
LOAD_MODEL = False
CKPT_FILEPATH = "ckpt.pt"


# hyperparameters
PREFERRED_DEVICE = "cuda"
T = 1000                                                                                                                 
EPOCHS = 10000
BATCH_SIZE = 12
LEARNING_RATE = 0.0003


# initialise logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# --------------------------------------
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
        try:
            schematic_path = self.schematics[idx]
            blocks, dimensions, _ = read_schematic(schematic_path) 
            blocks_tensor = transform_blocks(blocks, dimensions)

            return blocks_tensor, 0
        
        except Exception as e:
                print(f"Error processing schematic at index {idx}: {e}")
                return torch.zeros((1, 8, 8, 8)), 0


def transform_blocks(blocks, original_dimensions, target_dimensions=SCHEM_SHAPE):
    """
    transforms the 'blocks' array from a .schematic file to specified dimensions, and turns it into a tensor for pytorch

    Returns:
        torch.Tensor: Resized tensor of shape (1, D, H, W).
    """

    blocks_tensor = torch.tensor(blocks, dtype=torch.float) # convert to tensor
    blocks_tensor = torch.where(blocks_tensor == 0, torch.tensor(-1.0), torch.tensor(1.0)) # normalize values to -1 and 1 respectively    
    blocks_tensor = blocks_tensor.view(original_dimensions) # convert to 3d array [d*h*w] -> [D, H, W]
    blocks_tensor = blocks_tensor.unsqueeze(0).unsqueeze(0) # add channel & batch dimension [1, 1, D, H, W]

    blocks_tensor = F.interpolate(blocks_tensor, size=target_dimensions, mode='nearest') # resize [1, 1, (target_dimensions)]

    return blocks_tensor[0, :, :, :]

def create_schematic_from_tensor(filepath, blocks_tensor, dimensions=SCHEM_SHAPE, data=None):
    # ensure tensor is on cpu
    blocks = blocks_tensor.cpu().numpy()

    # remove batch and channel dimensions + convert to 1d array
    if len(blocks.shape) == 5 :
        blocks = blocks[0, 0, :, :, :].flatten()
    elif len(blocks.shape) == 4:
        blocks = blocks[0, :, :, :].flatten()

    if data is None: 
        data = np.zeros_like(blocks) 

    #ensure no invalid ids
    blocks = (blocks + 1) / 2
    blocks = np.where(np.round(blocks) == 1, 1, 0).astype(int)
    
    if np.prod(dimensions) != len(blocks):
        raise ValueError(f"Error creating schematic ({filepath}), invalid dimensions for the given blocks array")
        
    
    create_schematic(filepath, blocks, data, dimensions)



schem_dataset = SchematicsDataset(root="src/data/schematics_dataset") 
dataloader = DataLoader(schem_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)



# ---------------------------------------
# Forward Process - Noise Schedular
# ---------------------------------------

class NoiseSchedular:
    def __init__(self, beta_start=0.0001, beta_end=0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = PREFERRED_DEVICE

        # beta schedule 
        self.betas = self.linear_noise_schedule()

        # pre-calculate the necessary terms for noise calculations
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def linear_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, T).to(self.device)

    def noise_schematics(self, x, t):
        """
        Returns a noisy version of a schematic at a specific timestep 
        """
        sqrt_alpha_hat = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alphas_cumprod * noise, noise

    def sample_timesteps(self, n):
        """
        Returns random timestep(s)
        """
        return torch.randint(low=1, high=T, size=(n,))

    def sample(self, model, n=1):
        """
        Samples schematics using the model
        """
        logging.info(f"Sampling {n} new schematic(s)...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, *SCHEM_SHAPE)).to(self.device)
            for i in tqdm(reversed(range(1, T))):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alphas = self.alphas[t][:, None, None, None, None]
                alphas_cumprod = self.alphas_cumprod[t][:, None, None, None, None]
                betas = self.betas[t][:, None, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alphas_cumprod))) * predicted_noise) + torch.sqrt(betas) * noise
        model.train()

        x = torch.clamp(x, -1.0, 1.0) # normalise between -1 and 1
        return x


# --------------------------------------
# Backward Process - The U-Net
# ---------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, ch, size):
        super(SelfAttention, self).__init__()
        self.ch = ch
        self.size = size
        self.attention = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm([ch])
        self.feed_forward_sub = nn.Sequential(
            nn.LayerNorm([ch]),
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
        )

    def forward(self, x):
        
        # infer `self.size` dynamically
        spatial_dims = x.shape[2:]  # spatial dimensions (D, H, W)
        num_elements = spatial_dims[0] * spatial_dims[1] * spatial_dims[2]

        # reshape: (batch, channel, depth*height*width) -> (batch, num_elements, channel)
        x = x.view(-1, self.ch, num_elements).swapaxes(1, 2)

        # apply layer norm and attention
        x_ln = self.layer_norm(x)
        attention_value, _ = self.attention(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.feed_forward_sub(attention_value) + attention_value

        # reshape back to (batch, channel, depth, height, width)
        return attention_value.swapaxes(2, 1).view(-1, self.ch, *spatial_dims)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_ch
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, in_ch // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_ch
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, ch_in=1, ch_out=1, time_dim=256, device=PREFERRED_DEVICE):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # downsampling
        self.input = DoubleConv(ch_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # bottle neck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # upsampling
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.output = nn.Conv3d(64, ch_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.input(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.output(x)
        return output


# --------------------------------------
# Utility functions
# ---------------------------------------


def save_checkpoint(state, filepath):
    print("=> Saving checkpoint")
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    return start_epoch

def setup():
    os.makedirs("runs", exist_ok=True)
    os.makedirs(f"runs/{RUN_NAME}/models", exist_ok=True)
    os.makedirs(f"runs/{RUN_NAME}/results", exist_ok=True)



# --------------------------------------
# Training
# ---------------------------------------



device = torch.device(PREFERRED_DEVICE)

model = UNet()

if torch.cuda.device_count() > 1:
    logging.info(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

optimizer = optim.AdamW(model.parameters(), LEARNING_RATE)
mse = nn.MSELoss()
diffusion = NoiseSchedular()
writer = SummaryWriter(f"runs/{RUN_NAME}/logs")
l = len(dataloader)

# load an existing model
start_epoch = 0
if LOAD_MODEL: start_epoch = load_checkpoint(CKPT_FILEPATH, model, optimizer)




#create_schematic_from_tensor(f"runs/{RUN_NAME}/results/expected.schematic", next(iter(dataloader))[0])


if TRAIN:
    setup()
    for epoch in range(start_epoch, EPOCHS):
        logging.info(f"Epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (schematics, _) in enumerate(pbar):
            if torch.all(schematics == 0):
                continue

            schematics = schematics.to(device)
            t = diffusion.sample_timesteps(schematics.shape[0]).to(device)
            x_t, noise = diffusion.noise_schematics(schematics, t)

            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad() # reset previous gradients
            loss.backward() # compute the loss
            optimizer.step() # update the parameters
        
            pbar.set_postfix(Loss=loss.item())
            writer.add_scalar("Loss", loss.item(), global_step=epoch * l + i)

        if epoch % 100 == 0:
            sampled_schematics = diffusion.sample(model, 3)
     
            for sampled_schematic, i in enumerate(sampled_schematics):
                create_schematic_from_tensor(f"runs/{RUN_NAME}/results/{epoch}-{i}.schematic", sampled_schematic)

            state = {
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }

            save_checkpoint(state, f"runs/{RUN_NAME}/models/ckpt.pt")
    





