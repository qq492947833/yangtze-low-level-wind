"""Fit pointwise wind baselines, select alpha per lead and export native test forecasts."""
from pathlib import Path
import argparse
import json
import os
import tempfile

import numpy as np
from netCDF4 import Dataset
from threadpoolctl import threadpool_limits

import ridge_data as data
from boundary_math import merge_qr, solve_small, score_small
from ridge_math import DEFAULT_ALPHAS, SufficientStatistics, fit_ridge_path, validation_mse, select_global_alpha

EXTENDED_ALPHAS = np.array([1e-10, 1e-9, 1e-8, 1e-7])
WORK = Path(os.environ.get("YANGTZE_WORK_DIR", "work")).resolve() / "ridge"
RESULTS = Path(os.environ.get("YANGTZE_RESULTS_DIR", "results")).resolve()


def blocks(issues, point, lead, chunk=4096):
    for start in range(0, len(issues), chunk):
        sub = issues[start:start+chunk]
        yield data.feature_block(sub, [point])[0], data.target_block(sub, [point], lead)[0]


def moments(issues, point, lead):
    stats = SufficientStatistics.empty(data.N_FEATURES, 9)
    for x,y in blocks(issues, point, lead):
        stats.update(x,y)
    return stats


def compressed(issues, point, lead, path):
    r = None
    for x,y in blocks(issues, point, lead):
        x = (x-path.x_mean) / path.x_scale
        y = (y-path.y_mean) / path.y_scale
        r = merge_qr(r, np.column_stack((np.ones(len(x)),x,y)))
    return r


def fit_point(train, valid, point, lead, extended):
    path = fit_ridge_path(moments(train, point, lead))
    scores = validation_mse(path, moments(valid, point, lead))
    coef, bias = path.coef_std, path.intercept_std
    ols_score = None
    if extended:
        tr = compressed(train, point, lead, path)
        va = compressed(valid, point, lead, path)
        p = data.N_FEATURES + 1
        fit = solve_small(tr[:,1:p], tr[:,p:], len(train), np.r_[EXTENDED_ALPHAS,0.0], constant=tr[:,0])
        new = score_small(va[:,1:p], va[:,p:], va[:,0], fit["coef"], fit["intercept"], len(valid))
        ols_score = new[-1]
        scores = np.concatenate((new[:-1], scores))
        coef = np.concatenate((fit["coef"][:-1], coef))
        bias = np.concatenate((fit["intercept"][:-1], bias))
    arrays = dict(coef_std=coef, intercept_std=bias, x_mean=path.x_mean, x_scale=path.x_scale,
                  y_mean=path.y_mean, y_scale=path.y_scale)
    return arrays, scores, ols_score


def _predict_block(data,issues,model,npoints,point_block):
    result=np.empty((len(issues),9,npoints),dtype=np.float64)
    for start in range(0,npoints,point_block):
        stop=min(start+point_block,npoints);ids=np.arange(start,stop,dtype=np.int64)
        x=np.asarray(data.feature_block(issues,ids),dtype=np.float64)
        if x.shape!=(len(ids),len(issues),297) or not np.isfinite(x).all():raise ValueError('Invalid feature block')
        x=(x-model['x_mean'][start:stop,None,:])/model['x_scale'][start:stop,None,:]
        y=np.matmul(x,model['coef_std'][start:stop])+model['intercept_std'][start:stop,None,:]
        y=y*model['y_scale'][start:stop,None,:]+model['y_mean'][start:stop,None,:]
        if not np.isfinite(y).all():raise ValueError('Prediction block nonfinite')
        result[:,:,start:stop]=y.transpose(1,2,0)
    return result

def export(lead, model, selection, pred_dir):
    times = data.native_test(lead)
    co = data.coordinates()
    truth_dir = RESULTS / "ridge_baseline"
    truth_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for kind, folder in (("testy",truth_dir), ("predicty",pred_dir)):
        final = folder / f"Ridge_{lead}hour_{kind}.nc"
        tmp = final.with_suffix(".part.nc")
        with Dataset(tmp,"w") as ds:
            ds.alpha = selection["alpha"]
            ds.forecast_hours = lead
            for dim in ("time","level","latitude","longitude"):
                values = times["test_time_ns"] // 10**9 if dim=="time" else co[dim]
                ds.createDimension(dim, len(values))
                var = ds.createVariable(dim,"i8" if dim=="time" else "f8",(dim,))
                var[:] = values
                var.units = {"time":"seconds since 1970-01-01 00:00:00","level":"hPa",
                             "latitude":"degrees_north","longitude":"degrees_east"}[dim]
            variables = []
            for c in ("u","v","w"):
                var = ds.createVariable(f"{kind}_{c}","f8",("time","level","latitude","longitude"),zlib=True,complevel=4)
                var.units = "Pa s-1" if c=="w" else "m s-1"
                variables.append(var)
            for start in range(0,len(times["issues"]),32):
                issue = times["issues"][start:start+32]
                if kind=="predicty":
                    field = _predict_block(data,issue,model,data.N_POINTS,73)
                else:
                    field = data.target_block(issue,np.arange(data.N_POINTS),lead).transpose(1,2,0)
                field = field.reshape(len(issue),9,data.N_LAT,data.N_LON)
                for k,var in enumerate(variables):
                    var[start:start+len(issue)] = field[:,3*k:3*k+3]
        tmp.replace(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead",type=int,required=True,choices=range(1,7))
    parser.add_argument("--extended",action="store_true",help="Add the four smaller positive alphas and score OLS separately.")
    args = parser.parse_args()
    lead = args.lead
    alphas = np.r_[EXTENDED_ALPHAS,DEFAULT_ALPHAS] if args.extended else DEFAULT_ALPHAS
    folder = WORK / ("extended" if args.extended else "original") / f"lead_{lead}"
    folder.mkdir(parents=True, exist_ok=True)
    indices = data.load_indices()
    train = indices[f"h{lead}_train_issue_idx"]
    valid = indices[f"h{lead}_valid_issue_idx"]
    scores = np.empty((data.N_POINTS,len(alphas),9))
    ols = np.empty((data.N_POINTS,9)) if args.extended else None
    with tempfile.TemporaryDirectory(dir=folder) as temp, threadpool_limits(limits=2):
        for point in range(data.N_POINTS):
            model, scores[point], ols_score = fit_point(train,valid,point,lead,args.extended)
            if ols is not None:
                ols[point] = ols_score
            np.savez(Path(temp)/f"{point}.npz",**model)
        selection = select_global_alpha(scores,alphas)
        selection["candidate_alphas"] = alphas.tolist()
        if ols is not None:
            selection["ols_validation_mse"] = float(ols.mean())
        (folder/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
        model = {}
        for point in range(data.N_POINTS):
            with np.load(Path(temp)/f"{point}.npz") as z:
                for k in z.files:
                    value = z[k][selection["alpha_index"]] if k in ("coef_std","intercept_std") else z[k]
                    if k not in model:
                        model[k] = np.empty((data.N_POINTS,*value.shape),dtype=value.dtype)
                    model[k][point] = value
        np.savez(folder/"model.npz",alpha=selection["alpha"],**model)
    pred_dir = RESULTS / ("ridge_baseline_alpha_extended" if args.extended else "ridge_baseline")
    export(lead, model, selection, pred_dir)

if __name__=="__main__":
    main()
