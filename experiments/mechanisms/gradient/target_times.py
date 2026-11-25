"""Intersect the test target-time axes of the baseline and gradient variants at both stages."""
from common import np, xr, RESULT, ROOT, COMPONENTS, normalized_times


def main():
    common = None
    for h in range(1,7):
        for c in COMPONENTS:
            for suffix in ("", "_gradient", "_residual", "_residual_gradient"):
                for kind in ("testy","predicty"):
                    path = RESULT / f"da_transunet_{kind}_{c}_{h}_24year_changjiang{suffix}.nc"
                    with xr.open_dataset(path) as ds:
                        t = normalized_times(ds.time.values)
                    common = t if common is None else np.intersect1d(common,t,assume_unique=True)
    if len(common)==0:
        raise ValueError("No common test target hours")
    np.save(ROOT/"target_time.npy", common)

if __name__=="__main__":
    main()
