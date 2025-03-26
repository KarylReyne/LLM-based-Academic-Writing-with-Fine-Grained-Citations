import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from nn.attention import AttentionBlock, PatchEmbedding, ReversePatchEmbedding
from utils.loss import MaskedL1SSIMLoss as L1SSIMLoss
import numpy as np
from einops import rearrange, repeat, reduce

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        cfg,
    ):
        super(VisionTransformer, self).__init__()
        self.cfg = cfg

        self.loss = L1SSIMLoss()

        self.embedding = PatchEmbedding(cfg.num_frames, cfg.channels, patch_size=cfg.patch_size, image_height=cfg.image_height, image_width=cfg.image_width)
        
        self.layers = nn.Sequential(
            *[AttentionBlock(cfg.channels, cfg.num_heads) for _ in range(cfg.layers)],
        )
        
        self.reverse_embedding = ReversePatchEmbedding(cfg.channels, cfg.num_frames, patch_size=cfg.patch_size, image_height=cfg.image_height, image_width=cfg.image_width)
            
    def calculate_loss(self, x: th.Tensor, y: th.Tensor, patch_mask):
        patch_mask = rearrange(patch_mask, 'b (t h w) 1 -> b t h w', t=self.cfg.num_frames, h=self.cfg.image_height//self.cfg.patch_size, w=self.cfg.image_width//self.cfg.patch_size)
        patch_mask = repeat(patch_mask, 'b t h w -> b t (h h1) (w w1)', h1=self.cfg.patch_size, w1=self.cfg.patch_size)
        return self.loss(x, y, 1-patch_mask)

    def forward(self, input: th.Tensor):
        # Embed the input
        embedded, patch_mask = self.embedding(input)

        # Pass through the layers
        output = self.layers(embedded)

        # Reverse the embedding
        decoded = self.reverse_embedding(output)

        loss, l1, ssim = self.calculate_loss(decoded, input, patch_mask)
        return {'output': decoded, 'loss': loss, 'l1': l1, 'ssim': ssim}

