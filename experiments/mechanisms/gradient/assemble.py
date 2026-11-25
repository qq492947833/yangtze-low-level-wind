"""Assemble the gradient RMSE contrasts for Fig. 12 and supplementary figures."""
import argparse
import os
from pathlib import Path

import netCDF4
from common import ROOT, RESULT, np, xr

COMPONENTS = ("u", "v", "w")
SCORES = ROOT / "chunks" / "gradient"


def main(output):
  target = np.load(ROOT / "target_time.npy")
  base = ("stage","forecast_time","component")
  tail = ("condition","scale","level")
  gain = np.empty((2,6,3,7,3,3,4))
  ci_values = np.empty((2,6,3,3,2,7,3,3,4))
  draw = np.empty((2,6,3,3,1000,7,3,3))
  bnd = np.empty((2,6,3,7,3,3))
  rid_by_lead = {}
  threshold = None
  for si,stage in enumerate((1,2)):
    for hi,lead in enumerate(range(1,7)):
      for cc,component in enumerate(COMPONENTS):
        path = SCORES / f"h{lead}_{component}_s{stage}" / "result.nc"
        with xr.open_dataset(path) as case:
          np.testing.assert_array_equal(case.target_time.values,target)
          np.testing.assert_array_equal(case.block_hours.values,[72,24,168])
          np.testing.assert_array_equal(case.replicate.values,np.arange(1000))
          np.testing.assert_array_equal(case.level.values,[850,925,1000])
          np.testing.assert_array_equal(case.metric.values,["R","RMSE","MAE","RMSPE"])
          np.testing.assert_array_equal(case.condition.values,["all","spatial_low","spatial_middle","spatial_high","temporal_low","temporal_middle","temporal_high"])
          np.testing.assert_array_equal(case.scale.values,["original","box3_smooth","box3_remainder"])
          if lead in rid_by_lead:
            np.testing.assert_array_equal(rid_by_lead[lead],case.regime.values)
          else:
            rid_by_lead[lead]=case.regime.values
          if threshold is None:threshold=case.regime_threshold.values
          else:np.testing.assert_array_equal(threshold,case.regime_threshold.values)
          gain[si,hi,cc] = case.gain.values
          ci_values[si,hi,cc] = case.paired_gain_ci.values
          draw[si,hi,cc] = case.bootstrap_gain.sel(metric="RMSE").values
          # Each model's maximum reference perturbation bounds its RMSE error.
          eps = case.truth_reference_max_abs.sum("group").values
          bnd[si,hi,cc] = eps[None,None,:]
          bnd[si,hi,cc,:,2] *= 2
  output.parent.mkdir(parents=True, exist_ok=True)
  tmp = output.with_suffix(".part.nc")
  coords = dict(stage=[1,2],forecast_time=list(range(1,7)),component=list(COMPONENTS),
                condition=["all","spatial_low","spatial_middle","spatial_high","temporal_low","temporal_middle","temporal_high"],
                scale=["original","box3_smooth","box3_remainder"],level=[850,925,1000],
                metric=["R","RMSE","MAE","RMSPE"],block_hours=[72,24,168],bound=["lower","upper"],
                replicate=np.arange(1000),regime_axis=["spatial_gradient","time_change"])
  with netCDF4.Dataset(tmp,"w") as out:
   for k,values in coords.items():
    out.createDimension(k,len(values))
    values=np.asarray(values)
    dtype=str if values.dtype.kind in "OUS" else "i4"
    out.createVariable(k,dtype,(k,))[:] = values if dtype!=str else values.astype(object)
   def var(name,dtype,dims):
    return out.createVariable(name,dtype,dims,zlib=True,complevel=4)
   var("gain","f8",base+tail+("metric",))[:] = gain
   var("rmse_gain_bootstrap","f8",base+("block_hours","replicate")+tail)[:] = draw
   # Every statistic below has all times/conditions/scales/levels paired before derivation.
   change=gain[1,...,1]-gain[0,...,1];change_draw=draw[1]-draw[0];change_ci=np.nanquantile(change_draw,[.025,.975],axis=3).transpose(1,2,3,0,4,5,6);change_bound=bnd[0]+bnd[1];expanded=change_ci.copy();expanded[:,:,:,0]-=change_bound[:,:,None];expanded[:,:,:,1]+=change_bound[:,:,None]
   cd=('forecast_time','component')+tail;cid=('forecast_time','component','block_hours','bound')+tail
   var('rmse_gain_change_stage2_minus_stage1','f8',cd)[:]=change;var('rmse_gain_change_ci_saved_reference','f8',cid)[:]=change_ci;var('rmse_gain_change_ci_raw_bound_expanded','f8',cid)[:]=expanded;var('rmse_gain_change_raw_truth_bound','f8',cd)[:]=change_bound
   var('stage_rmse_gain_ci_raw_bound_expanded','f8',base+('block_hours','bound')+tail)
   sci=ci_values[...,1].copy();sci[:,:,:,:,0]-=bnd[:,:,:,None];sci[:,:,:,:,1]+=bnd[:,:,:,None];out['stage_rmse_gain_ci_raw_bound_expanded'][:]=sci
   # Conditional high-low contrast uses the saved paired draw order; pressure levels stay separate.
   hl=np.stack([gain[:,:,:,3,...,1]-gain[:,:,:,1,...,1],gain[:,:,:,6,...,1]-gain[:,:,:,4,...,1]],axis=3)
   hd=np.stack([draw[:,:,:,:,:,3]-draw[:,:,:,:,:,1],draw[:,:,:,:,:,6]-draw[:,:,:,:,:,4]],axis=5)
   hci=np.nanquantile(hd,[.025,.975],axis=4).transpose(1,2,3,4,0,5,6,7)
   var('rmse_gain_condition_high_minus_low','f8',base+('regime_axis','scale','level'))[:]=hl
   var('rmse_gain_condition_high_minus_low_ci','f8',base+('block_hours','bound','regime_axis','scale','level'))[:]=hci
   var('rmse_gain_condition_high_minus_low_raw_truth_bound','f8',base+('regime_axis','scale','level'))[:]=np.repeat((2*bnd[:,:,:,0])[:,:,:,None],2,axis=3)
   out.setncatts(dict(stage1_definition="Main-model wind prediction",stage2_definition="Complete residual-corrected wind",
                     bootstrap="Paired nonoverlapping whole-time blocks, 1000 replicates; primary 72 h and sensitivity 24/168 h",
                     gain_units="RMSE in m s-1 for u and v, Pa s-1 for omega"))
  os.replace(tmp,output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=RESULT/"task2_diagnostics/mechanism_v4_20261028/gradient/gradient_mechanism_v4.nc")
    main(parser.parse_args().output)
