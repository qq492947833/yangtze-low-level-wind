"""Transform 100 background and 100 sample physical 11-hour histories into native stage inputs."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "zwd"))
from model_definition import build_model
from expected_gradients import configure_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True,help="NPZ, background_values and x_values [100,11,3,49,73], sp/t2m/d2m.")
    parser.add_argument("--weights",type=Path,required=True,help="Stage 1 trained state dictionary for the selected lead.")
    parser.add_argument("--normalization",type=Path,required=True,help="Same lead's two-stage residual normalization NPZ.")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    device = torch.device(os.environ.get("YANGTZE_DEVICE","cuda:0"))
    configure_gpu(device)
    network = build_model(3,"cpu")
    network.load_state_dict(torch.load(args.weights,map_location="cpu",weights_only=True))
    network.eval().to(device)
    with np.load(args.normalization) as f:
        stats = {k:torch.from_numpy(f[k].transpose(2,0,1).copy()).float().to(device)
                 for k in ("vx_mean","vx_std","vy_mean","vy_std","prediction_input_mean","prediction_input_std")}
    arrays = {}
    with np.load(args.raw) as raw, torch.inference_mode():
        for group,key in [("background","background_values"),("sample","x_values")]:
            x = raw[key]
            if x.shape!=(100,11,3,49,73):
                raise ValueError(f"{key} must be [100,11,3,49,73]")
            stage1, stage2 = [], []
            for start in range(0,len(x),2):
                a = torch.from_numpy(x[start:start+2]).float().to(device)
                xn = (a-stats["vx_mean"])/stats["vx_std"]
                windows = torch.stack([xn[:,k:k+6] for k in range(6)],dim=1).flatten(0,1)
                layout = windows.permute(0,1,3,4,2).contiguous().permute(0,1,4,2,3)
                bn = network(layout).reshape(len(a),6,9,49,73)
                bp = (bn.double()*stats["vy_std"].double()+stats["vy_mean"].double()).float()
                pin = (bp-stats["prediction_input_mean"])/stats["prediction_input_std"]
                stage1.append(xn[:,5:].cpu().numpy())
                stage2.append(torch.cat([xn[:,5:],pin],dim=2).cpu().numpy())
            arrays["stage1_"+group] = np.concatenate(stage1)
            arrays["stage2_"+group] = np.concatenate(stage2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)


if __name__ == "__main__":
    main()
