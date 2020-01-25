from scipy.stats import norm, rankdata
import numpy as np
import pandas as pd
import pickle
from pystan import StanModel
import falcon
from falcon_cors import CORS
import pprint
from json import dumps, load


pp = pprint.PrettyPrinter(indent=4)


def get_percentiles(x):
    x_n = (1 - x) / 2
    return { (str(x * 100) + '%'): [x_n, 1 - x_n] }

class TTest:
    def on_post(self, req, resp, **kwargs):
        raw_data = load(req.bounded_stream)['params']
        print(raw_data)
        y1 = pd.Series(raw_data['y1'])
        y0 = pd.Series(raw_data['y0'])
        y_comb = pd.concat([y1, y0])
        x_comb = pd.concat(
            [pd.Series([1] * y1.size), pd.Series([0] * y0.size)])
        dat_list = {
            'y': y_comb, 'x': x_comb, 'N': y1.size + y0.size,
            'sd_m': raw_data['sd_m'],
            'sd_m_diff': raw_data['max_diff'] / norm.ppf(.975),
            'sd_st': raw_data['sd_st'],
            'sd_st_r': np.log(raw_data['max_st_r']) / norm.ppf(.975),
            'nu_choice': raw_data['nu_choice']
        }
        params = {new_list: {} for new_list in [
            'm0', 'm1', 'm_diff', 'st0', 'st1', 'st_ratio', 'nu', 'ps']}
        dmdv = pickle.load(open('stan_scripts/dmdv.pkl', 'rb'))
        fit = dmdv.sampling(data=dat_list, chains=4, iter=raw_data['n_iter'])

        # ('mean', 'se_mean', 'sd', 'n_eff', 'Rhat')

        summary = fit.summary(pars=params.keys(), probs=[])['summary']

        posteriors = fit.extract()
        for i in range(len(params)):
            param = list(params.keys())[i]
            params[param] = {
                'mean': summary[i][0], 'median': np.median(posteriors[param]),
                'mcse': summary[i][1], 'sd': summary[i][2],
                'post': posteriors[param].tolist(), 'ranks': rankdata(posteriors[param]).tolist(),
                # 'quantiles': np.quantile(posteriors[param], list(map(lambda x: x / 200, range(1, 200)))).tolist(),
                'quantiles': np.quantile(posteriors[param], list(map(lambda x: x / 1000, range(1, 1000)))).tolist(),
                'ess': summary[i][3], 'rhat': summary[i][4],
                # todo: 'warnings': 
            }
        
        params['raw_data'] = raw_data

        # print(params)
        # pp.pprint(params)
        resp.media = params

cors = CORS(allow_all_origins=True, allow_all_methods=True, allow_all_headers=True)
api = falcon.API(middleware=[cors.middleware])

# api = falcon.API()
api.add_route('/two_sample_test', TTest())
