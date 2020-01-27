from scipy.stats import norm, rankdata
import numpy as np
import pandas as pd
import pickle
from pystan import StanModel
import falcon
from falcon_cors import CORS
import pprint
from json import dumps, load
from base64 import b64encode, b64decode
import io
import matplotlib.pyplot as plt
import arviz as az


pp = pprint.PrettyPrinter(indent=4)


def get_percentiles(x):
    x_n = (1 - x) / 2
    return {(str(x * 100) + '%'): [x_n, 1 - x_n]}


def img_b64(bytes):
    bytes.seek(0)
    b64_hash = b64encode(bytes.read())
    return b64_hash.decode('utf-8')


def create_qty_plt(samples, ref):
    y = list(
        map(lambda x: x / 1000, range(1, 1000)))
    x = np.quantile(samples, y)
    y = y[::-1]
    ytks = [.001, .5, .8, .95, .999]
    xtks = np.quantile(samples, [.001, .05, .2, .5, .999])
    left, width = 0.15, .83
    bottom, height = 0.13, 0.7
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.15]
    plt.figure(figsize=(5, 6))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.tick_params(axis='x', labelrotation=30)
    ax_scatter.plot(x, y, color='black')
    ax_scatter.set_yticks(ytks)
    ax_scatter.set_yticklabels(['{:,.1%}'.format(x)
                                for x in ax_scatter.get_yticks()])
    ax_scatter.set_xticks(xtks)
    ax_scatter.set_xlabel('threshold')
    ax_scatter.set_ylabel('P(statistic > threshold)')
    for ytk in ytks:
        ax_scatter.axhline(ytk, color="#5d5d5d", linestyle=':')
    for xtk in xtks:
        ax_scatter.axvline(xtk, color="#5d5d5d", linestyle=':')
    if not ((ref is None) or (ref > np.max(samples)) or (ref < np.min(samples))):
        ax_scatter.axvline(ref, color="#5d5d5d", linestyle='--')
    ax_histx = plt.axes(rect_histx)
    ax_histx.axis('off')
    ax_histx.hist(x, bins=30, color='#5d5d5d')
    ax_histx.set_xlim(ax_scatter.get_xlim())
    plt_IOBytes = io.BytesIO()
    plt.savefig(plt_IOBytes, format='png')
    return img_b64(plt_IOBytes)


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
            'm0', 'm1', 'm_diff', 'st0', 'st1', 'st_ratio', 'nu']}
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
                'post': posteriors[param].tolist(), # 'ranks': rankdata(posteriors[param]).tolist(),
                'ess': summary[i][3], 'rhat': summary[i][4],
                # todo: 'warnings': 
            }

        rk_IOBytes = io.BytesIO()
        rank_plot = az.plot_rank(fit, var_names = ('m_diff', 'st_ratio'))
        plt.savefig(rk_IOBytes, format='png')
        params['rk_hash'] = img_b64(rk_IOBytes)

        params['mean_hash'] = create_qty_plt(posteriors['m_diff'], 0)
        params['sc_hash'] = create_qty_plt(posteriors['st_ratio'], 1)

        params['raw_data'] = raw_data

        # print(params)
        # pp.pprint(params)
        resp.media = params

cors = CORS(allow_all_origins=True, allow_all_methods=True, allow_all_headers=True)
api = falcon.API(middleware=[cors.middleware])

# api = falcon.API()
api.add_route('/two_sample_test', TTest())
