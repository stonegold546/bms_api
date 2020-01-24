from scipy.stats import norm
import numpy as np
import pickle
from pystan import StanModel
import matplotlib.pyplot as plt
import arviz as az

dat_list = {
    'y': [
        101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103, 100, 95, 102, 106,
        109, 102, 82, 102, 100, 102, 102, 101, 102, 102, 103, 103, 97, 97, 103, 101, 97, 104,
        96, 103, 124, 101, 101, 100, 101, 101, 104, 100, 101,
        99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105, 88, 101, 100,
        104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100, 101, 99, 101, 100, 100,
        101, 100, 99, 101, 100, 102, 99, 100, 99
        ],
    'x': np.concatenate(([1] * 47, [0] * 42)),
    'sd_m': 5, 'sd_m_diff': 15 / norm.ppf(.975),
    'sd_st': 1, 'sd_st_r': np.log(3) / norm.ppf(.975),
    'nu_choice': 0
}
dat_list['N'] = len(dat_list['y'])

# dmdv = StanModel(file='stan_scripts/dmdv.stan')
# with open('stan_scripts/dmdv.pkl', 'wb') as f:
#     pickle.dump(dmdv, f)
# ddfdmdv = StanModel(file='stan_scripts/ddfdmdv.stan')
# with open('stan_scripts/ddfdmdv.pkl', 'wb') as f:
#     pickle.dump(ddfdmdv, f)

dmdv = pickle.load(open('stan_scripts/dmdv.pkl', 'rb'))
# ddfdmdv = pickle.load(open('stan_scripts/ddfdmdv.pkl', 'rb'))

fit = dmdv.sampling(data=dat_list, chains=4, iter=4000)
# print(fit)

az.plot_rank(fit, var_names=['m_diff', 'st_ratio', 'nu']); plt.show()
az.plot_posterior(fit, var_names=['m_diff', 'st_ratio', 'nu'],
                  credible_interval=.95, kind='hist', point_estimate='median')
plt.show()
# posteriors = fit.extract()
# posteriors['m_diff']
# az.plot_trace(fit, var_names=['m_diff'])
# plt.show()
# az.plot_dist(posteriors['m_diff'])
# plt.show()
