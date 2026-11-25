import numpy as np

def frozen_draws(seed,nsamples,background_count):
    public=np.random.RandomState(seed);rseed=int(public.randint(0,1000000));rng=np.random.RandomState(rseed)
    indices=np.empty(nsamples,np.int64);alpha=np.empty(nsamples,np.float64)
    for k in range(nsamples):indices[k]=rng.choice(background_count);alpha[k]=rng.uniform()
    return {'effective_rseed':rseed,'background_index':indices,'alpha':alpha}
