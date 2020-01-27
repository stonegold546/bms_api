from scipy.stats import norm
import numpy as np
import pickle
from pystan import StanModel

dat_list = {
    'y': [101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103, 100, 95, 102, 106,
        109, 102, 82, 102, 100, 102, 102, 101, 102, 102, 103, 103, 97, 97, 103, 101, 97, 104,
        96, 103, 124, 101, 101, 100, 101, 101, 104, 100, 101,
        99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105, 88, 101, 100,
        104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100, 101, 99, 101, 100, 100,
        101, 100, 99, 101, 100, 102, 99, 100, 99],
    'x': np.concatenate(([1] * 47, [0] * 42)),
    'N': 89, 'sd_m': 5, 'sd_m_diff': 15 / norm.ppf(.975),
    'sd_st': 1, 'sd_st_r': 3 / norm.ppf(.975), 'nu_choice': 0
}

sm = StanModel(file='dmdv.stan')
fit = sm.sampling(data=dat_list, iter=2000, chains=4)
print(fit)

# save it to the file 'model.pkl' for later use
with open('dmdv.pkl', 'wb') as f:
    pickle.dump(sm, f)
