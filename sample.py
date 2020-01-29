from scipy.stats import norm
import numpy as np
import pandas as pd
import pickle
import falcon
from falcon_cors import CORS
import pprint
from json import load
from base64 import b64encode
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


def create_qty_plt(samples, name, percent=False):
    y = list(
        map(lambda x: x / 1000, range(1, 1000)))
    x = np.quantile(samples, y)
    y = y[::-1]
    ytks = [.001, .5, .8, .95, .999]
    xtks = np.quantile(samples, [.001, .05, .2, .5, .999])
    left, width = 0.15, .8
    bottom, height = 0.13, 0.65
    spacing = 0.05
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
    ax_scatter.set_ylabel('P(' + name + ' > threshold)')
    for ytk in ytks:
        ax_scatter.axhline(ytk, color="#5d5d5d", linestyle=':')
    for xtk in xtks:
        ax_scatter.axvline(xtk, color="#5d5d5d", linestyle=':')
    # if not ((ref is None) or (ref > np.max(samples)) or (ref < np.min(samples))):
    #     ax_scatter.axvline(ref, color="#5d5d5d", linestyle='--')
    ax_histx = plt.axes(rect_histx)
    ax_histx.get_yaxis().set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.hist(x, bins=30, color='#5d5d5d', ec='black')
    ax_histx.set_xlim(ax_scatter.get_xlim())
    if percent:
        ax_scatter.set_xticklabels(['{:,.2%}'.format(x)
                                    for x in ax_scatter.get_xticks()])
    plt_IOBytes = io.BytesIO()
    plt.savefig(plt_IOBytes, format='png')
    plt.close()
    return img_b64(plt_IOBytes)


class TTest:
    def on_post(self, req, resp, **kwargs):
        raw_data = load(req.bounded_stream)['params']
        # print(raw_data)
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
        fit = dmdv.sampling(data=dat_list, chains=4,
                            iter=raw_data['n_iter'], seed=12345)

        # ('mean', 'se_mean', 'sd', 'n_eff', 'Rhat')

        summary = fit.summary(pars=params.keys(), probs=[])['summary']

        posteriors = fit.extract()
        for i in range(len(params)):
            param = list(params.keys())[i]
            params[param] = {
                'mean': summary[i][0], 'median': np.median(posteriors[param]),
                'mcse': summary[i][1], 'sd': summary[i][2],
                'post': posteriors[param].tolist(),
                'ess': summary[i][3], 'rhat': summary[i][4],
                # todo: 'warnings':
            }

        rk_IOBytes = io.BytesIO()
        az.plot_rank(fit, var_names=('m_diff', 'st_ratio'))
        plt.savefig(rk_IOBytes, format='png')
        plt.close()
        params['rk_hash'] = img_b64(rk_IOBytes)

        params['mean_hash'] = create_qty_plt(
            posteriors['m_diff'], 'mean difference')
        params['sc_hash'] = create_qty_plt(
            posteriors['st_ratio'], 'SD ratio')

        params['raw_data'] = raw_data

        resp.media = params


class MetaRE:
    def on_post(self, req, resp, **kwargs):
        raw_data = load(req.bounded_stream)['params']
        eff = pd.Series(raw_data['eff'])
        scales = pd.Series(raw_data['scales'])
        dat_list = {
            'eff': eff, 's': scales, 'N': eff.size,
            'mean_sd': raw_data['mean_sd'] / norm.ppf(.975),
            'rnd_sd': raw_data['rnd_sd'] / norm.ppf(.975)
        }
        params = {new_list: {}
                  for new_list in ['eff_mean', 'eff_sd', 'eff_model']}
        meta_re = pickle.load(open('stan_scripts/meta_re.pkl', 'rb'))
        fit = meta_re.sampling(data=dat_list, chains=4,
                               iter=raw_data['n_iter'], seed=12345)
        lo_prob = (1 - raw_data['interval'] / 100) / 2
        hi_prob = 1 - lo_prob

        summary = fit.summary(pars=params.keys(), probs=(
            lo_prob, .5, hi_prob))['summary']

        posteriors = fit.extract()
        for i in range(len(params)):
            param = list(params.keys())[i]
            if param == 'eff_model':
                for j in range(raw_data['N']):
                    par_sum = summary[i + j]
                    params[param][param + '[' + str(j + 1) + ']'] = {
                        'mean': par_sum[0], 'median': par_sum[4],
                        'mcse': par_sum[1], 'sd': par_sum[2],
                        'int.lo': par_sum[3], 'int.hi': par_sum[5],
                        'ess': par_sum[6], 'rhat': par_sum[7]
                    }
            else:
                par_sum = summary[i]
                params[param] = {
                    'mean': par_sum[0], 'median': par_sum[4],
                    'mcse': par_sum[1], 'sd': par_sum[2],
                    'post': posteriors[param].tolist(),
                    'int.lo': par_sum[3], 'int.hi': par_sum[5],
                    'ess': par_sum[6], 'rhat': par_sum[7]
                }

        rk_IOBytes = io.BytesIO()
        az.plot_rank(fit, var_names=('eff_mean', 'eff_sd'))
        plt.savefig(rk_IOBytes, format='png')
        plt.close()
        params['rk_hash'] = img_b64(rk_IOBytes)

        params['mean_hash'] = create_qty_plt(
            posteriors['eff_mean'], 'average effect')
        params['sd_hash'] = create_qty_plt(
            posteriors['eff_sd'], 'tau')

        params['raw_data'] = raw_data

        resp.media = params


class TwoGroupBin:
    def on_post(self, req, resp, **kwargs):
        raw_data = load(req.bounded_stream)['params']
        success = pd.Series(raw_data['success'])
        total = pd.Series(raw_data['total'])
        dat_list = {
            'pass': success, 'total': total,
            'sd_m': raw_data['sd_m'] / norm.ppf(.975),
            'sd_m_diff': np.log(raw_data['sd_m_diff']) / norm.ppf(.975)
        }
        params = {new_list: {}
                  for new_list in ['m0', 'm_diff', 'means_logit[2]',
                                   'odds_ratio', 'means_prob[1]', 'means_prob[2]',
                                   'prob_ratio', 'prob_diff']}
        two_group_logistic = pickle.load(open('stan_scripts/two_group_logistic.pkl', 'rb'))
        fit = two_group_logistic.sampling(data=dat_list, chains=4,
                                          iter=raw_data['n_iter'], seed=12345)
        lo_prob = (1 - raw_data['interval'] / 100) / 2
        hi_prob = 1 - lo_prob

        summary = fit.summary(pars=params.keys(), probs=(
            lo_prob, .5, hi_prob))['summary']

        posteriors = fit.extract()
        for i in range(len(params)):
            param = list(params.keys())[i]
            par_sum = summary[i]
            if param == 'means_logit[2]':
                param = 'm1'
            params[param] = {
                'mean': par_sum[0], 'median': par_sum[4],
                'mcse': par_sum[1], 'sd': par_sum[2],
                'int.lo': par_sum[3], 'int.hi': par_sum[5],
                'ess': par_sum[6], 'rhat': par_sum[7]
            }
            if param in ['m0', 'm_diff', 'odds_ratio', 'prob_ratio', 'prob_diff']:
                params[param]['post'] = posteriors[param].tolist()

        params.pop('means_logit[2]')
        rk_IOBytes = io.BytesIO()
        az.plot_rank(fit, var_names=('odds_ratio'))
        plt.savefig(rk_IOBytes, format='png')
        plt.close()
        params['rk_hash'] = img_b64(rk_IOBytes)

        params['or_hash'] = create_qty_plt(
            posteriors['odds_ratio'], 'odds ratio')
        params['pr_hash'] = create_qty_plt(
            posteriors['prob_ratio'], 'prob. ratio')
        params['pd_hash'] = create_qty_plt(
            posteriors['prob_diff'], 'prob. diff')

        params['raw_data'] = raw_data

        resp.media = params


cors = CORS(allow_all_origins=True,
            allow_all_methods=True, allow_all_headers=True)
api = application = falcon.API(middleware=[cors.middleware])

api.add_route('/two_sample_test', TTest())
api.add_route('/meta_re', MetaRE())
api.add_route('/two_sample_binary', TwoGroupBin())
