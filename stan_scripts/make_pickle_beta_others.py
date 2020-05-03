from scipy.stats import norm
import numpy as np
import pickle
from pystan import StanModel

sm_bt = StanModel(file='dmdv_beta.stan')
with open('dmdv_beta.pkl', 'wb') as f:
    pickle.dump(sm_bt, f)
