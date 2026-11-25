"""Regional-mean expected gradients for one DA-TransUNet stage and forecast lead."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "zwd"))
from model_definition import build_model
from expected_gradients import configure_gpu
from quality import integrate_outputs
from stagewise_wrapper import StageRegionalWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True,help="Prepared NPZ with stage1/2_background and stage1/2_sample.")
    parser.add_argument("--weights",type=Path,required=True,help="Stage-specific trained state dictionary.")
    parser.add_argument("--normalization",type=Path,required=True,help="Residual normalization NPZ containing the stage output mean/std.")
    parser.add_argument("--mask",type=Path,help="Optional matching regional mask NPZ. The prepared inputs already carry it.")
    parser.add_argument("--lead",type=int,choices=range(1,7),required=True)
    parser.add_argument("--stage",type=int,choices=(1,2),required=True)
    parser.add_argument("--sample",type=int,choices=range(100),required=True)
    parser.add_argument("--max-draws",type=int,choices=(4096,8192,16384,32768,65536,131072,262144),default=262144)
    parser.add_argument("--microbatch",type=int,default=2)
    parser.add_argument("--blocks",type=Path,required=True,help="Directory of converged sample blocks.")
    args = parser.parse_args()
    device = torch.device(os.environ.get("YANGTZE_DEVICE","cuda:0"))
    configure_gpu(device)
    network = build_model(3 if args.stage==1 else 12,"cpu")
    network.load_state_dict(torch.load(args.weights,map_location="cpu",weights_only=True))
    with np.load(args.mask or args.inputs) as f:
        mask = f["output_valid_mask"].astype(bool)
    with np.load(args.normalization) as f:
        name = "vy" if args.stage==1 else "residual"
        mean,std = (f[f"{name}_{key}"].transpose(2,0,1).copy() for key in ("mean","std"))
    model = StageRegionalWrapper(network,mean,std,mask,args.stage).eval().to(device)
    with np.load(args.inputs) as f:
        if not np.array_equal(f["output_valid_mask"], mask):
            raise ValueError("Fixed stage mask mismatch")
        init_time = f["sample_init_time"][args.sample]
        valid_time = f["sample_valid_time"][args.sample, args.lead - 1]
        pool_index = f["sample_pool_index"][args.sample]
        bg = torch.from_numpy(f[f"stage{args.stage}_background"])
        x = torch.from_numpy(f[f"stage{args.stage}_sample"][args.sample:args.sample+1])
    with torch.no_grad():
        preds = model(x.to(device))[:,:,0,0].cpu().numpy()[0]
        model.cleanup()
        expected = np.zeros(9,dtype=np.float64)
        for pair in bg.split(2):
            for row in model(pair.to(device))[:,:,0,0].cpu().numpy():
                expected += np.asarray(row,dtype=np.float64)
            model.cleanup()
        expected /= len(bg)
    factor = np.array([1.]*6+[-10.]*3)[:,None,None]
    scale = (np.abs(factor*std.astype(np.float64))*mask).sum((1,2))/mask.sum((1,2))
    values = integrate_outputs(model,x,bg,preds,expected,scale,args.stage,args.lead,args.sample,
                               args.microbatch,args.max_draws)
    args.blocks.mkdir(parents=True,exist_ok=True)
    out = args.blocks / f"stage{args.stage}_lead{args.lead}_sample{args.sample:03d}.npz"
    tmp = out.with_suffix(".pending.npz")
    np.savez_compressed(tmp,**values,sample=args.sample,stage=args.stage,lead=args.lead,
                        init_time=init_time,valid_time=valid_time,pool_index=pool_index)
    tmp.replace(out)


if __name__ == "__main__":
    main()
