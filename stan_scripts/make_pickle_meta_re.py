from scipy.stats import norm
import pickle
from pystan import StanModel

# 8 schools
dat_list = {
    'eff': [28,  8, -3,  7, -1,  1, 18, 12],
    's': [15, 10, 16, 11,  9, 11, 10, 18],
    'N': 8, 'mean_sd': 15 / norm.ppf(.975),
    'rnd_sd': 20 / norm.ppf(.975)
}

sm = StanModel(file='meta_re.stan')
fit = sm.sampling(data=dat_list, iter=2000, chains=4, seed=12345)
print(fit)

with open('meta_re.pkl', 'wb') as f:
    pickle.dump(sm, f)
