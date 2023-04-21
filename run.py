import os
import numpy as np
import pylab as plt
import pickle
from Util import createObservations, CR, interpolate
from DA import dataAssimilation

import matplotlib as mpl

# plt.rc('text', usetex=True)
# plt.rc('font', family='times', size=12)
# plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')
rng = np.random.default_rng(6)

path_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'


# ======================================================================================================================
# ======================================================================================================================
def main(filter_ens, truth, method, results_dir="results/", save_=False):
    os.makedirs(results_dir, exist_ok=True)

    # =========================  PERFORM DATA ASSIMILATION ========================== #
    filter_ens = dataAssimilation(filter_ens, truth['p_obs'], truth['t_obs'],
                                  std_obs=truth['std_obs'], method=method)
    # Integrate further without assimilation as ensemble mean (if truth very long, integrate only .2s more)
    Nt_extra = 0
    if filter_ens.hist_t[-1] < truth['t'][-1]:
        Nt_extra = int(min((truth['t'][-1] - filter_ens.hist_t[-1]), filter_ens.t_CR) / filter_ens.dt) + 1
        psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
        filter_ens.updateHistory(psi, t)
        if filter_ens.bias is not None:
            y = filter_ens.getObservableHist(Nt_extra)
            b, t_b = filter_ens.bias.timeIntegrate(t=t, y=y)
            filter_ens.bias.updateHistory(b, t_b)
    filter_ens.close()

    # ================================== SAVE DATA  ================================== #
    parameters = dict(biasType=filter_ens.biasType, forecast_model=filter_ens.name,
                      true_model=truth['model'], num_DA=len(truth['t_obs']), Nt_extra=Nt_extra)
    # filter_ens = filter_ens.getOutputs()
    if save_:
        filename = '{}{}-{}_F-{}'.format(results_dir, method, truth['name'], filter_ens.name)
        if filter_ens.bias.name == 'ESN':
            filename += '_k{}'.format(filter_ens.bias.k)

        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
            pickle.dump(truth, f)
            pickle.dump(filter_ens, f)

    return filter_ens, truth, parameters


# ======================================================================================================================
# ======================================================================================================================
def createEnsemble(true_p, forecast_p, filter_p, bias_p,
                   working_dir="results", filename='reference_Ensemble', results_dir=None):
    if results_dir is None:
        results_dir = working_dir

    if os.path.isfile(results_dir + filename):
        with open(results_dir + filename, 'rb') as f:
            ensemble = pickle.load(f)
            truth = pickle.load(f)
            b_args = pickle.load(f)
        reinit = False
        # check that true and forecast model parameters
        for key, val in filter_p.items():
            if hasattr(ensemble, key) and getattr(ensemble, key) != val:
                reinit = True
                print('Re-initialise ensemble as ensemble {}={} != {}'.format(key, getattr(ensemble, key), val))
                break
            elif b_args is not None and type(b_args) is dict and key in b_args[0].keys() and val != b_args[0][key]:
                reinit = True
                print('Re-initialise ensemble as filter_p {}={} != {}'.format(key, b_args[0][key], val))
                break
        if truth['t_obs'][-1] < filter_p['t_stop']:
            reinit = True

        if not reinit and bias_p is not None:
            # check that bias and assimilation parameters are the same
            for key, val in bias_p.items():
                if key in b_args[0]['Bdict'].keys():
                    try:
                        if val != b_args[0]['Bdict'][key]:
                            reinit = True
                            print('Re-init ensemble as {} = {} != {}'.format(key, b_args[0]['Bdict'][key], val))
                            break
                    except:
                        for v1, v2 in zip(val, b_args[0]['Bdict'][key]):
                            if v1 != v2:
                                reinit = True
                                print('Re-init ensemble as {} = {} != {}'.format(key, b_args[0]['Bdict'][key], val))
                                break
        if not reinit:
            # Remove transient to save up space
            i_transient = np.argmin(abs(truth['t'] - ensemble.t_transient))
            for key in ['y', 't', 'b']:
                truth[key] = truth[key][i_transient:]
            return ensemble, truth, b_args

    # =============================  CREATE OBSERVATIONS ============================== #
    y_true, t_true, name_truth = createObservations(true_p)

    if 'manual_bias' in true_p.keys():
        if true_p['manual_bias'] == 'time':
            b_true = .4 * y_true * np.sin((np.expand_dims(t_true, -1) * np.pi * 2) ** 2)
        elif true_p['manual_bias'] == 'periodic':
            b_true = 0.2 * np.max(y_true, 0) * np.cos(2 * y_true / np.max(y_true, 0))
        elif true_p['manual_bias'] == 'linear':
            b_true = .1 * np.max(y_true, 0) + .3 * y_true
        elif true_p['manual_bias'] == 'cosine':
            b_true = np.cos(y_true)
        else:
            raise ValueError("Bias type not recognised choose: 'linear', 'periodic', 'time'")
    else:
        b_true = np.zeros(1)

    y_true += b_true
    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(round(filter_p['t_start'] / dt_t),
                        round(filter_p['t_stop'] / dt_t) + 1, filter_p['kmeas'])
    t_obs = t_true[obs_idx]
    q = np.shape(y_true)[1]
    if 'std_obs' not in true_p.keys():
        true_p['std_obs'] = 0.01
    Cdd = np.eye(q) * true_p['std_obs'] ** 2

    noise = rng.multivariate_normal(np.zeros(q), Cdd, len(obs_idx))
    obs = y_true[obs_idx] * (1. + noise)
    truth = dict(y=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_obs, p_obs=obs, dt_obs=t_obs[1] - t_obs[0],
                 true_params=true_p, name=name_truth,
                 model=true_p['model'], std_obs=true_p['std_obs'])

    # %% =============================  DEFINE BIAS ======================================== #
    if filter_p['biasType'].name == 'ESN':
        args = (filter_p, forecast_p['model'], truth, working_dir)
        filter_p['Bdict'] = createESNbias(*args, bias_param=bias_p)
    else:
        args = (None,)

    # ===============================  INITIALISE ENSEMBLE  =============================== #
    ensemble = forecast_p['model'](forecast_p, filter_p)
    with open(results_dir + filename, 'wb') as f:
        pickle.dump(ensemble, f)
        pickle.dump(truth, f)
        pickle.dump(args, f)

    return ensemble, truth, args


# ======================================================================================================================
# ======================================================================================================================
def createESNbias(filter_p, model, truth, folder, bias_param=None):
    os.makedirs(folder, exist_ok=True)

    if bias_param is None:
        return dict()

    if 'L' not in bias_param.keys():
        bias_param['L'] = 10

    bias_p = bias_param.copy()
    train_params = bias_p['train_params'].copy()
    train_params['m'] = bias_p['L']

    # Compute reference bias. Create an ensemble of training data
    ref_ens = model(train_params, train_params)
    try:
        name_train = folder + 'Truth_{}_{}'.format(ref_ens.name, ref_ens.law)
    except:
        name_train = folder + 'Truth_{}'.format(ref_ens.name)

    for k in ref_ens.params:
        name_train += '_{}{}'.format(k, getattr(ref_ens, k))
    name_train += '_std{:.2}_m{}_{}'.format(ref_ens.std_a, ref_ens.m, ref_ens.alpha_distr)
    # Load or create reference ensemble (multi-parameter solution)
    rerun = True
    print(name_train)
    if os.path.isfile(name_train):
        with open(name_train, 'rb') as f:
            load_ens = pickle.load(f)
        if len(truth['t']) <= len(load_ens.hist_t):
            ref_ens = load_ens.copy()
            rerun = False
    if rerun:
        print('Creating Reference solution(s)')
        psi, t = ref_ens.timeIntegrate(Nt=len(truth['t']) - 1)
        ref_ens.updateHistory(psi, t)
        ref_ens.close()
        with open(name_train, 'wb') as f:
            pickle.dump(ref_ens, f)

    y_ref, lbl = ref_ens.getObservableHist(Nt=len(truth['t'])), ref_ens.obsLabels
    t = ref_ens.hist_t[:len(truth['t'])]

    if len(truth['y'].shape) < len(y_ref.shape):
        biasData = np.expand_dims(truth['y'], -1) - y_ref  # [Nt x Nmic x L]
    else:
        biasData = truth['y'] - y_ref  # [Nt x Nmic x L]

    # provide data for washout before first observation
    if 'start_ensemble_forecast' not in filter_p.keys():
        filter_p['start_ensemble_forecast'] = 2

    tol = 1e-5
    i1 = truth['t_obs'][0] - truth['dt_obs'] * filter_p['start_ensemble_forecast']
    i1 = int(np.where(abs(truth['t'] - i1) < tol)[0])
    i0 = i1 - bias_p['N_wash'] * bias_p['upsample']
    if i0 < 0:
        min_t = (bias_p['N_wash'] * bias_p['upsample'] + filter_p['kmeas']) * (t[1] - t[0])
        raise ValueError('increase t_start to > t_wash + dt_a = {}'.format(min_t))

    # create bias dictionary
    bias_p['trainData'] = biasData
    bias_p['washout_obs'] = truth['y'][i0:i1 + 1]
    bias_p['washout_t'] = truth['t'][i0:i1 + 1]
    bias_p['filename'] = folder + truth['name'] + '_' + name_train.split('Truth_')[-1] + '_bias'

    # Plot training data -------------------------------------
    fig, axs = plt.subplots(figsize=(12, 4.5), layout='constrained')
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1])
    axs_top = subfigs[0].subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 2]})
    axs_bot = subfigs[1].subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 2]})
    norm = mpl.colors.Normalize(vmin=-5, vmax=y_ref.shape[-1] + 2)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.magma)

    Nt = int(ref_ens.t_CR / truth['dt'])
    L = y_ref.shape[-1]

    i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))
    i0_r = np.argmin(abs(ref_ens.hist_t - truth['t_obs'][0]))

    yt = truth['y'][i0_t - Nt:i0_t]
    bt = truth['b'][i0_t - Nt:i0_t]
    yr = y_ref[i0_r - Nt:i0_r]
    tt = ref_ens.hist_t[i0_r - Nt:i0_r]

    axs_top[0].plot(tt, yt[:, 0], color='silver', linewidth=6, alpha=.8)
    axs_top[-1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.8)
    axs_bot[0].plot(t, truth['b'][:, 0], color='silver', linewidth=4, alpha=.8)

    xlims = [[truth['t_obs'][0] - ref_ens.t_CR, truth['t_obs'][0]],
             [truth['t_obs'][0], truth['t_obs'][0] + ref_ens.t_CR * 2]]

    for ii in range(y_ref.shape[-1]):
        clr = cmap.to_rgba(y_ref.shape[-1] - ii)
        C, R = CR(yt, yr[:, :, ii])
        axs_top[0].plot(tt, yr[:, 0, ii], color=clr)
        axs_bot[1].plot(ii, C, 'o', color=clr, markersize=4)
        axs_top[1].plot(ii, R, 'x', color=clr, markersize=4)
        norm_bias = (truth['y'][:, 0] - y_ref[:, 0, ii])
        axs_bot[-1].plot(t, norm_bias, color=clr)
        axs_top[-1].plot(t, norm_bias, color=clr)

    axs_top[0].legend(['Truth'], bbox_to_anchor=(0., 0.25), loc="upper left")
    axs_top[0].set(xlabel='$t$', ylabel='$y$', xlim=xlims[0])
    axs_bot[1].set(xlabel='$l$', ylabel='Correlation')
    axs_top[1].set(xlabel='$l$', ylabel='RMS error')
    axs_top[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[0])
    axs_bot[0].set(xlabel='$t$', ylabel='$b$', xlim=xlims[-1])
    axs_bot[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[-1])
    axs_top[0].plot(truth['t'], truth['y'][:, 0], color='silver', linewidth=6, alpha=.8)
    for ax_ in [axs_bot[1], axs_top[1]]:
        x0, x1 = ax_.get_xlim()
        y0, y1 = ax_.get_ylim()
        ax_.set_aspect((x1 - x0) / (y1 - y0))

    os.makedirs(path_dir + folder, exist_ok=True)
    plt.savefig(path_dir + folder + 'L{}_training_data.svg'.format(L), dpi=350)
    plt.close()
    return bias_p


# ======================================================================================================================
# ======================================================================================================================
def get_error_metrics(results_folder):
    print('computing error metrics...')
    out = dict(Ls=[], ks=[])

    L_dirs, k_files = [], []
    LLL = os.listdir(results_folder)
    for Ldir in LLL:
        if os.path.isdir(results_folder + Ldir + '/') and Ldir[0] == 'L':
            L_dirs.append(results_folder + Ldir + '/')
            out['Ls'].append(float(Ldir.split('L')[-1]))

    for ff in os.listdir(L_dirs[0]):
        k = float(ff.split('_k')[-1])
        out['ks'].append(k)
        k_files.append(ff)

    # sort ks and Ls
    idx_ks = np.argsort(np.array(out['ks']))
    out['ks'] = np.array(out['ks'])[idx_ks]
    out['k_files'] = [k_files[i] for i in idx_ks]

    idx = np.argsort(np.array(out['Ls']))
    out['L_dirs'] = [L_dirs[i] for i in idx]
    out['Ls'] = np.array(out['Ls'])[idx]

    # Output quantities
    keys = ['R_biased_DA', 'R_biased_post',
            'C_biased_DA', 'C_biased_post',
            'R_unbiased_DA', 'R_unbiased_post',
            'C_unbiased_DA', 'C_unbiased_post']
    for key in keys:
        out[key] = np.empty([len(out['Ls']), len(out['ks'])])

    print(out['Ls'])
    print(out['ks'])

    from plotResults import post_process_single
    ii = -1
    for Ldir in out['L_dirs']:
        ii += 1
        print('L = ', out['Ls'][ii])
        jj = -1
        for ff in out['k_files']:
            jj += 1
            # Read file
            with open(Ldir + ff, 'rb') as f:
                _ = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
            truth = truth.copy()

            print('\t k = ', out['ks'][jj], '({}, {})'.format(filter_ens.bias.L, filter_ens.bias.k))
            # Compute biased and unbiased signals
            y, t = filter_ens.getObservableHist(), filter_ens.hist_t
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
            y_mean = np.mean(y, -1)

            # Unbiased signal error
            if hasattr(filter_ens.bias, 'upsample'):
                y_unbiased = y_mean[::filter_ens.bias.upsample] + b
                y_unbiased = interpolate(t_b, y_unbiased, t)
            else:
                y_unbiased = y_mean + b

            # if jj == 0:
            N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
            i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
            i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation

            # cut signals to interval of interest
            y_mean, t, y_unbiased = y_mean[i0 - N_CR:i1 + N_CR], t[i0 - N_CR:i1 + N_CR], y_unbiased[i0 - N_CR:i1 + N_CR]

            if ii == 0 and jj == 0:
                i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
                i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                y_truth, t_truth = truth['y'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
                y_truth_b = y_truth - truth['b'][i0_t - N_CR:i1_t + N_CR]

                out['C_true'], out['R_true'] = CR(y_truth[-N_CR:], y_truth_b[-N_CR:])
                out['C_pre'], out['R_pre'] = CR(y_truth[:N_CR], y_mean[:N_CR])
                out['t_interp'] = t[::N_CR]
                scale = np.max(y_truth, axis=0)
                for key in ['error_biased', 'error_unbiased']:
                    out[key] = np.empty([len(out['Ls']), len(out['ks']), len(out['t_interp']), y_mean.shape[-1]])

            # End of assimilation
            for yy, key in zip([y_mean, y_unbiased], ['_biased_DA', '_unbiased_DA']):
                C, R = CR(y_truth[-N_CR * 2:-N_CR], yy[-N_CR * 2:-N_CR])
                out['C' + key][ii, jj] = C
                out['R' + key][ii, jj] = R

            # After Assimilaiton
            for yy, key in zip([y_mean, y_unbiased], ['_biased_post', '_unbiased_post']):
                C, R = CR(y_truth[-N_CR:], yy[-N_CR:])
                out['C' + key][ii, jj] = C
                out['R' + key][ii, jj] = R

            # Compute mean errors
            b_obs = y_truth - y_mean
            b_obs_u = y_truth - y_unbiased
            ei, a = -N_CR, -1
            while ei < len(b_obs) - N_CR - 1:
                a += 1
                ei += N_CR
                out['error_biased'][ii, jj, a, :] = np.mean(abs(b_obs[ei:ei + N_CR]), axis=0) / scale
                out['error_unbiased'][ii, jj, a, :] = np.mean(abs(b_obs_u[ei:ei + N_CR]), axis=0) / scale

    with open(results_folder + 'CR_data', 'wb') as f:
        pickle.dump(out, f)

# def plot_train_data(truth, y_ref, t_CR, folder):
#     # Plot training data -------------------------------------
#     fig, ax = plt.subplots(1, 3, figsize=(15, 3.5), layout='constrained')
#     norm = mpl.colors.Normalize(vmin=-5, vmax=y_ref.shape[-1])
#     cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.magma)
#     fig.suptitle('Training data')
#     ax[0].plot(truth['t'], truth['y'][:, 0], color='silver', linewidth=6, alpha=.8)
#     Nt = int(t_CR / truth['dt'])
#
#     L = y_ref.shape[-1]
#     for ii in range(y_ref.shape[-1]):
#         C, R = CR(truth['y'][-Nt:], y_ref[-Nt:, :, ii])
#         line = ax[0].plot(truth['t'], y_ref[:, 0, ii], color=cmap.to_rgba(ii))
#         ax[1].plot(ii, C, 'o', color=cmap.to_rgba(ii))
#         ax[2].plot(ii, R, 'x', color=cmap.to_rgba(ii))
#     ax[0].legend(['Truth'], bbox_to_anchor=(0., 1.25), loc="upper left")
#     ax[0].set(xlabel='$t$', ylabel='y', xlim=[truth['t'][-1] - t_CR, truth['t'][-1]])
#     ax[1].set(xlabel='$l$', ylabel='Correlation')
#     ax[2].set(xlabel='$l$', ylabel='RMS error')
#     ax[0].plot(truth['t'], truth['y'][:, 0], color='silver', linewidth=6, alpha=.8)
#     plt.savefig(folder + 'L{}_training_data.svg'.format(L), dpi=350)
#     plt.close()
