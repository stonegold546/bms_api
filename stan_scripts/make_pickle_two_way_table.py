from scipy.stats import norm
import numpy as np
import pickle
from pystan import StanModel

# progesterone
dat_list = {
    'pass': [1459, 1513], 'total': [2013, 2025],
    'sd_m': 4 / norm.ppf(.975),
    'sd_m_diff': np.log(2) / norm.ppf(.975)
}

sm = StanModel(file='two_group_logistic.stan')
fit = sm.sampling(data=dat_list, iter=2000, chains=4, seed=12345)
print(fit)

with open('two_group_logistic.pkl', 'wb') as f:
    pickle.dump(sm, f)
