"""Physical per-channel spatial means inside forward, before attribution."""
import numpy as np
import torch
from torch import nn
def physical_region_mean(raw_output,vy_mean,vy_std,output_mask):
    if raw_output.ndim!=4 or raw_output.shape[1]!=9:raise ValueError('raw_output[B,9,H,W] required')
    mean=torch.as_tensor(vy_mean,device=raw_output.device,dtype=torch.float64)
    std=torch.as_tensor(vy_std,device=raw_output.device,dtype=torch.float64)
    mask=torch.as_tensor(output_mask,device=raw_output.device,dtype=torch.bool)
    if mean.shape!=raw_output.shape[1:] or std.shape!=mean.shape or mask.shape!=mean.shape:raise ValueError('C,H,W stats/mask mismatch')
    counts=mask.sum((-2,-1))
    if bool((counts==0).any()) or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):raise ValueError('invalid mask/statistics')
    factors=torch.ones((9,1,1),dtype=torch.float64,device=raw_output.device);factors[6:]=-10.
    physical=(raw_output.double()*std+mean)*factors
    return torch.where(mask.unsqueeze(0),physical,torch.zeros_like(physical)).sum((-2,-1))/counts
def aggregate_features(element_phi):
    phi=np.asarray(element_phi,dtype=np.float64)
    if phi.ndim!=5 or (phi.shape[1]!=6 or phi.shape[2] not in (3,12)) or not np.isfinite(phi).all():raise ValueError('phi[K,6,3_or_12,H,W] required')
    return phi.sum(axis=(1,3,4))
class StageRegionalWrapper(nn.Module):
    """Direct frozen stage network; X is its actual normalized input."""
    def __init__(self,network,output_mean,output_std,output_mask,stage):
        super().__init__();self.network=network;self.stage=int(stage)
        if self.stage not in (1,2):raise ValueError('stage1 or stage2')
        self.register_buffer('output_mean',torch.as_tensor(output_mean).clone())
        self.register_buffer('output_std',torch.as_tensor(output_std).clone())
        self.register_buffer('output_mask',torch.as_tensor(output_mask,dtype=torch.bool).clone())
        if self.output_mean.shape!=(9,49,73) or self.output_std.shape!=self.output_mean.shape or self.output_mask.shape!=self.output_mean.shape:raise ValueError('stats/mask mismatch')
        self.eval()
    def forward(self,x):
        if x.ndim!=5 or tuple(x.shape[1:])!=(6,3 if self.stage==1 else 12,49,73):raise ValueError('actual normalized stage input required')
        # Restore the original channels-last storage layout, not a new preprocessing.
        layout=x.permute(0,1,3,4,2).contiguous().permute(0,1,4,2,3)
        raw=self.network(layout)
        return physical_region_mean(raw,self.output_mean,self.output_std,self.output_mask)[:,:,None,None]
    def cleanup(self):
        for k in list(self.network.__dict__):
            if k.startswith('model_') or k.startswith('covup') or k.startswith('covdown') or k in ('x','res_layer'):self.network.__dict__.pop(k,None)
