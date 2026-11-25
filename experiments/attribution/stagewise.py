"""Regional-mean expected gradients for one DA-TransUNet stage and forecast lead."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "zwd"))
from model_definition import build_model
from expected_gradients import configure_gpu, gpu_expected_gradients
from stagewise_wrapper import StageRegionalWrapper, aggregate_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True,help="Prepared NPZ with stage1/2_background and stage1/2_sample.")
    parser.add_argument("--weights",type=Path,required=True,help="Stage-specific trained state dictionary.")
    parser.add_argument("--normalization",type=Path,required=True,help="Residual normalization NPZ containing the stage output mean/std.")
    parser.add_argument("--mask",type=Path,required=True,help="NPZ with output_valid_mask [9,49,73].")
    parser.add_argument("--stage",type=int,choices=(1,2),required=True)
    parser.add_argument("--sample",type=int,choices=range(100),required=True)
    parser.add_argument("--draws",type=int,default=2048)
    parser.add_argument("--microbatch",type=int,default=2)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    device = torch.device(os.environ.get("YANGTZE_DEVICE","cuda:0"))
    configure_gpu(device)
    network = build_model(3 if args.stage==1 else 12,"cpu")
    network.load_state_dict(torch.load(args.weights,map_location="cpu",weights_only=True))
    with np.load(args.mask) as f:
        mask = f["output_valid_mask"].astype(bool)
    with np.load(args.normalization) as f:
        name = "vy" if args.stage==1 else "residual"
        mean,std = (f[f"{name}_{key}"].transpose(2,0,1).copy() for key in ("mean","std"))
    model = StageRegionalWrapper(network,mean,std,mask,args.stage).eval().to(device)
    with np.load(args.inputs) as f:
        bg = torch.from_numpy(f[f"stage{args.stage}_background"])
        x = torch.from_numpy(f[f"stage{args.stage}_sample"][args.sample:args.sample+1])
    with torch.no_grad():
        preds = model(x.to(device))[:,:,0,0].cpu().numpy()[0]
        model.cleanup()
        baseline = []
        for pair in bg.split(2):
            baseline.append(model(pair.to(device))[:,:,0,0].cpu().numpy())
            model.cleanup()
        expected = np.concatenate(baseline).mean(axis=0)
    phi,_ = gpu_expected_gradients(model,x,bg,[(c,0,0) for c in range(9)],
        seed=25+7919*args.sample,nsamples=args.draws,microbatch=args.microbatch)
    scale = (np.abs(np.array([1.]*6+[-10.]*3)[:,None,None]*std)*mask).sum((1,2))/mask.sum((1,2))
    error = phi.reshape(9,-1).sum(1)-(preds-expected)
    tolerance = .02*scale+.05*np.maximum(np.abs(preds-expected),.1*scale)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,phi_feature=aggregate_features(phi),prediction=preds,
                        expected_value=expected,efficiency_error=error,efficiency_tolerance=tolerance,
                        closure_pass=np.abs(error)<=tolerance,nsamples=args.draws,sample=args.sample,stage=args.stage)


if __name__ == "__main__":
    main()
