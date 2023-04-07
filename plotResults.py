import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
from Util import interpolate, CR, getEnvelope
from scipy.interpolate import CubicSpline, interp1d, interp2d
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from run import get_error_metrics

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


def post_process_loopParams(results_dir, k_plot=(None,), figs_dir=None, k_max=100.):
    if results_dir[-1] != '/':
        results_dir += '/'
    if figs_dir is None:
        figs_dir = results_dir + 'figs/'
    os.makedirs(figs_dir, exist_ok=True)

    std_dirs = os.listdir(results_dir)
    for item in std_dirs:
        if not os.path.isdir(results_dir + item) or item[:3] != 'std':
            continue
        print(results_dir + item)
        std_folder = results_dir + item + '/'
        std = item.split('std')[-1]

        # Plot contours
        filename = '{}Contour_std{}_results'.format(figs_dir, std)
        plot_Lk_contours(std_folder, filename)

        # Plot CR and means
        filename = '{}CR_std{}_results'.format(figs_dir, std)
        post_process_multiple(std_folder, filename, k_max=k_max)
        #
        # Plot timeseries
        if k_plot is not None:
            L_dirs = os.listdir(std_folder)
            for L_item in L_dirs:
                L_folder = std_folder + L_item + '/'
                if not os.path.isdir(L_folder) or L_item[0] != 'L':
                    print(L_folder)
                    continue
                L = L_item.split('L')[-1]
                for k_item in os.listdir(L_folder):
                    kval = float(k_item.split('_k')[-1])
                    if kval in k_plot:
                        with open(L_folder + k_item, 'rb') as f:
                            params = pickle.load(f)
                            truth = pickle.load(f)
                            filter_ens = pickle.load(f)
                        # filename = '{}L{}_std{}_k{}_time'.format(figs_dir, L, std, kval)
                        # print(filename)
                        # post_process_single_SE_Zooms(filter_ens, truth, filename=filename)
                        filename = '{}L{}_std{}_k{}_J'.format(figs_dir, L, std, kval)
                        post_process_single(filter_ens, truth, params, filename=filename)


def post_process_WhyAugment(results_dir, k_plot=None, L_plot=None, figs_dir=None):
    if figs_dir is None:
        figs_dir = results_dir + 'figs/'
    os.makedirs(figs_dir, exist_ok=True)

    flag = True
    xtags, mydirs = [], []
    for Ldir in sorted(os.listdir(results_dir), key=str.lower):
        if not os.path.isdir(results_dir + Ldir + '/') or len(Ldir.split('_Augment')) == 1:
            continue
        mydirs.append(Ldir)

    k_files = []
    ks = []
    for ff in os.listdir(results_dir + mydirs[0] + '/'):
        k = float(ff.split('_k')[-1])
        if k_plot is not None and k not in k_plot:
            continue
        k_files.append(ff)
        ks.append(k)
    # sort ks and Ls
    idx_ks = np.argsort(np.array(ks))
    ks = np.array(ks)[idx_ks]
    k_files = [k_files[i] for i in idx_ks]

    cols = mpl.colormaps['viridis'](np.linspace(0., 1., len(ks)*2))

    barData = [[] for _ in range(len(ks) * 2)]

    for Ldir in mydirs:
        values = Ldir.split('_L')[-1]
        print(Ldir.split('_Augment'))
        L, augment = values.split('_Augment')
        if augment == 'True':
            augment = True
        else:
            augment = False
        L = int(L.split('L')[-1])
        if L_plot is not None and L not in L_plot:
            continue

        xtags.append('$L={}$'.format(L))
        if augment:
            xtags[-1] += '\n \\& data augment'
        ii = -2
        for ff in k_files:
            with open(results_dir + Ldir + '/' + ff, 'rb') as f:
                params = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)

            ii += 2
            truth = truth.copy()
            # ---------------------------------------------------------
            y, t = filter_ens.getObservableHist(), filter_ens.hist_t
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

            # Unbiased signal error
            if hasattr(filter_ens.bias, 'upsample'):
                y_unbiased = y[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
                y_unbiased = interpolate(t_b, y_unbiased, t)
            else:
                y_unbiased = y + b

            N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
            i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
            i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation

            # cut signals to interval of interest
            y, t, y_unbiased = y[i0-N_CR:i1+N_CR], t[i0-N_CR:i1+N_CR], y_unbiased[i0-N_CR:i1+N_CR]
            y_mean = np.mean(y, -1)

            if flag and ii == 0:
                i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
                i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                y_truth, t_truth = truth['y'][i0_t-N_CR:i1_t+N_CR], truth['t'][i0_t-N_CR:i1_t+N_CR]
                y_truth_b = y_truth - truth['b'][i0_t - N_CR:i1_t + N_CR]

                Ct, Rt = CR(y_truth[-N_CR:], y_truth_b[-N_CR:])
                Cpre, Rpre = CR(y_truth[:N_CR], y_mean[:N_CR:])


            # GET CORRELATION AND RMS ERROR =====================================================================
            CB, RB, CU, RU = [np.zeros(y.shape[-1]) for _ in range(4)]
            for mi in range(y.shape[-1]):
                CB[mi], RB[mi] = CR(y_truth[-N_CR:], y[-N_CR:, :, mi])  # biased
                CU[mi], RU[mi] = CR(y_truth[-N_CR:], y_unbiased[-N_CR:, :, mi])  # unbiased

            barData[ii].append((np.mean(CU), np.mean(RU), np.std(CU), np.std(RU)))
            barData[ii + 1].append((np.mean(CB), np.mean(RB), np.std(CB), np.std(RB)))


            if filter_ens.bias.k in k_plot:
                filename = '{}WhyAugment_L{}_augment{}_k{}'.format(figs_dir, L, augment, filter_ens.bias.k)
                # print(filename)
                # post_process_single_SE_Zooms(filter_ens, truth, filename=filename)
                post_process_single(filter_ens, truth, params, filename=filename + '_J')

        flag = False

    # --------------------------------------------------------- #
    # cols = ['b', 'c', 'r', 'coral', 'g', '#c1fd95', 'k', 'gray', '#a87900', '#fbdd7e']

    labels = []
    for kk in ks:
        labels.append('$\\gamma = {}$, Unbiased'.format(kk))
        labels.append('$\\gamma = {}$, Biased'.format(kk))

    bar_width = 0.1
    bars = [np.arange(len(barData[0]))]
    for _ in range(len(ks) * 2):
        bars.append([x + bar_width for x in bars[-1]])

    fig, ax = plt.subplots(1, 2, figsize=(12, 3), layout="constrained")

    for data, br, c, lb in zip(barData, bars, cols, labels):
        C = np.array([x[0] for x in data]).T.squeeze()
        R = np.array([x[1] for x in data]).T.squeeze()
        Cstd = np.array([x[2] for x in data]).T.squeeze()
        Rstd = np.array([x[3] for x in data]).T.squeeze()
        ax[0].bar(br, C, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[0].errorbar(br, C, yerr=Cstd, fmt='o', capsize=2., color='k', markersize=2)
        ax[1].bar(br, R, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[1].errorbar(br, R, yerr=Rstd, fmt='o', capsize=2., color='k', markersize=2)

    for axi, cr in zip(ax, [(Ct, Cpre), (Rt, Rpre)]):
        axi.axhline(y=cr[0], color='lightgray', linewidth=4, label='Truth')
        axi.axhline(y=cr[1], color='k', linewidth=2, label='Pre-DA')
        axi.set_xticks([r + bar_width for r in range(len(data))], xtags)

    ax[0].set(ylabel='Correlation', ylim=[.85, 1.02])
    ax[1].set(ylabel='RMS error', ylim=[0, Rpre * 1.5])
    # axi.legend(bbox_to_anchor=(1., 1.), loc="upper left")

    plt.savefig(figs_dir + 'WhyAugment.svg', dpi=350)
    # plt.savefig(figs_dir + 'WhyAugment.pdf', dpi=350)
    plt.close()


# ==================================================================================================================
def post_process_single_SE_Zooms(ensemble, true_data, filename=None):
    truth = true_data.copy()
    filter_ens = ensemble.copy()

    t_obs, obs = truth['t_obs'], truth['p_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    # %% ================================ PLOT time series, parameters and RMS ================================ #

    y_filter, labels = filter_ens.getObservableHist(), filter_ens.obsLabels
    if len(np.shape(y_filter)) < 3:
        y_filter = np.expand_dims(y_filter, axis=1)
    if len(np.shape(truth['y'])) < 3:
        truth['y'] = np.expand_dims(truth['y'], axis=-1)

    # normalise results

    hist = filter_ens.hist
    t = filter_ens.hist_t

    mean = np.mean(hist, -1, keepdims=True)
    y_mean = np.mean(y_filter, -1)
    std = np.std(y_filter[:, 0, :], axis=1)

    fig, ax = plt.subplots(1, 3, figsize=[17.5, 3.5], layout="constrained")
    params_ax, zoomPre_ax, zoom_ax = ax[:]
    x_lims = [t_obs[0] - .05, t_obs[-1] + .05]

    c = 'lightgray'
    zoom_ax.plot(truth['t'], truth['y'][:, 0], color=c, linewidth=8)
    zoomPre_ax.plot(truth['t'], truth['y'][:, 0], color=c, label='Truth', linewidth=8)
    zoomPre_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='black', linewidth=.8)
    zoom_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='black', linewidth=.8)

    zoom_ax.plot(t_obs, obs[:, 0], '.', color='r', markersize=10)
    zoomPre_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observations', markersize=10)
    if filter_ens.bias is not None:
        c = 'navy'
        b = filter_ens.bias.hist
        t_b = filter_ens.bias.hist_t

        if filter_ens.bias.name == 'ESN':
            y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            spline = interp1d(t_b, y_unbiased, kind='cubic', axis=0, copy=True, bounds_error=False, fill_value=0)
            y_unbiased = spline(t)

            t_wash = filter_ens.bias.washout_t
            wash = filter_ens.bias.washout_obs

            zoomPre_ax.plot(t_wash, wash[:, 0], '.', color='r', markersize=10)
            washidx = np.argmin(abs(t - filter_ens.bias.washout_t[0]))
        else:
            y_unbiased = y_filter + np.expand_dims(b, -1)
            washidx = np.argmin(abs(t - filter_ens.bias.hist_t[0]))

        y_mean_u = np.mean(y_unbiased, -1)
        zoom_ax.plot(t, y_mean_u[:, 0], '-', label='Unbiased filtered signal', color=c, linewidth=1.5)
        zoomPre_ax.plot(t[washidx:], y_mean_u[washidx:, 0], '-', color=c, linewidth=1.5)

    c = 'lightseagreen'
    zoom_ax.plot(t, y_mean[:, 0], '--', color=c, label='Biased estimate', linewidth=1.5, alpha=0.9)
    zoomPre_ax.plot(t, y_mean[:, 0], '--', color=c, linewidth=1.5, alpha=0.9)
    zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoomPre_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)

    zoomPre_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    zoom_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    y_lims = [min(min(y_mean[:, 0]), min(truth['y'][:, 0])) * 1.2,
              max(max(y_mean[:, 0]), max(truth['y'][:, 0])) * 1.2]
    zoom_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[-1] - 0.03, t_obs[-1] + 0.02], ylim=y_lims)
    zoomPre_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[0] - 0.03, t_obs[0] + 0.02], ylim=y_lims)

    # PLOT PARAMETER CONVERGENCE-------------------------------------------------------------
    ii = len(filter_ens.psi0)
    c = ['g', 'mediumpurple', 'sandybrown', 'cyan']
    params_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
    params_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='dimgray')

    if num_DA_blind > 0:
        params_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue',
                       label='Start BE')
    if num_SE_only > 0:
        params_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet',
                       label='Start PE')

    if filter_ens.est_p:
        max_p, min_p = -np.infty, np.infty
        for p in filter_ens.est_p:
            superscript = '^\mathrm{init}$'
            reference_p = filter_ens.alpha0

            mean_p = mean[:, ii].squeeze() / reference_p[p]
            std = np.std(hist[:, ii] / reference_p[p], axis=1)

            max_p = max(max_p, max(mean_p))
            min_p = min(min_p, min(mean_p))

            if p in ['C1', 'C2']:
                params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$' + p + '/' + p + superscript)
            else:
                params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$\\' + p + '/\\' + p + superscript)

            params_ax.set(xlabel='$t$ [s]', xlim=x_lims)
            params_ax.fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[ii - len(filter_ens.psi0)])
            ii += 1
        params_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=3)
        params_ax.plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)
        params_ax.set(ylim=[min_p - 0.1, max_p + 0.1])

    if filename is not None:
        # plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
        plt.close()
    else:
        plt.show()


def post_process_single(filter_ens, truth, params, filename=None, CR_file=None):
    filt = filter_ens.filt
    biasType = filter_ens.biasType
    Nt_extra = params['Nt_extra']

    t_obs, obs = truth['t_obs'], truth['p_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    y_filter, t = filter_ens.getObservableHist(), filter_ens.hist_t
    b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

    y_mean = np.mean(y_filter, -1)

    if hasattr(filter_ens.bias, 'upsample'):
        y_unbiased = y_mean[::filter_ens.bias.upsample] + b
        y_unbiased = interpolate(t_b, y_unbiased, t)
    else:
        y_unbiased = y_mean + b

    # cut signals to interval of interest -----
    N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS

    i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
    i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation
    y_filter, y_mean, t, y_unbiased = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_filter, y_mean, t, y_unbiased])

    y_truth = interpolate(truth['t'], truth['y'], t)
    std = np.std(y_filter[:, 0, :], axis=1)

    # %% PLOT time series ------------------------------------------------------------------------------------------

    fig, ax = plt.subplots(3, 3, figsize=[17.5, 10], layout="constrained")
    fig.suptitle(filter_ens.name + ' DA with ' + filt + ' and ' + biasType.name + ' bias')
    p_ax, zoomPre_ax, zoom_ax = ax[0, :]
    params_ax, bias_ax, biaszoom_ax = ax[1, :]
    RMS_ax, J_ax, dJ_ax = ax[2, :]

    labels = [('Truth', 'Observation data', '', ''),
              ('', '', 'Unbiased filtered signal', ''),
              ('', '', '', 'Biased filtered signal')]
    # y_lims = [min(min(y_truth[:, 0]), min(y_mean[:, 0])) * 1.1,
    #           max(max(y_truth[:, 0]), max(y_mean[:, 0])) * 1.1]
    y_lims = [np.min(y_truth[:N_CR, 0]) * 1.1,
              np.max(y_truth[:N_CR, 0]) * 1.1]
    x_lims = [[t[0], t[-1]],
              [t[0], t[0] + 2 * filter_ens.t_CR],
              [t[-1] - 2 * filter_ens.t_CR, t[-1]]]

    for ax_, lbl, xl in zip([p_ax, zoomPre_ax, zoom_ax], labels, x_lims):
        ax_.plot(t, y_truth[:, 0], color='lightgray', label=lbl[0], linewidth=8)
        ax_.plot(t, y_unbiased[:, 0], '-', color='navy', label=lbl[2], linewidth=1.5)
        ax_.plot(t, y_mean[:, 0], '--', color='lightseagreen', linewidth=1.5, alpha=0.9, label=lbl[3])
        ax_.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color='lightseagreen')
        ax_.plot(t_obs, obs[:, 0], '.', color='r', label=lbl[1], markersize=10)
        ax_.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
        ax_.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
        ax_.set(xlabel='$t$ [s]', ylim=y_lims, xlim=xl)
        ax_.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    p_ax.set(ylabel="$p'_\mathrm{mic_1}$ [Pa]")

    # BIAS PLOTS ------------------------------------------------------------------------------------------------
    if filter_ens.bias is not None:
        b = filter_ens.bias.hist
        t_b = filter_ens.bias.hist_t

        if filter_ens.bias.name == 'ESN':
            t_wash = filter_ens.bias.washout_t
            wash = filter_ens.bias.washout_obs
            p_ax.plot(t_wash, wash[:, 0], '.', color='r')
            zoomPre_ax.plot(t_wash, wash[:, 0], '.', color='r', markersize=10)

        b_obs = y_truth - y_mean
        y_lims_b = [np.min(b_obs[:N_CR, 0]) * 1.1, np.max(b_obs[:N_CR, 0]) * 1.1]
        for b_ax, x_lim in zip([bias_ax, biaszoom_ax], x_lims[1:]):
            b_ax.plot(t, b_obs[:, 0], color='darkorchid', label='Observable bias', alpha=0.4, linewidth=6)
            b_ax.plot(t_b, b[:, 0], color='darkorchid', label='ESN estimation')
            b_ax.set(ylabel='Bias', xlabel='$t$ [s]', ylim=y_lims_b, xlim=x_lim)
            b_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
            b_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
            b_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window

    # PLOT PARAMETER CONVERGENCE-------------------------------------------------------------
    hist, hist_t = filter_ens.hist, filter_ens.hist_t
    hist_mean = np.mean(hist, -1, keepdims=True)
    x_lims = [t[0], t[-1]]
    ii = len(filter_ens.psi0)
    ii0 = len(filter_ens.psi0)
    cs = ['g', 'sandybrown', 'mediumpurple', 'cyan']
    params_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
    params_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='dimgray')
    if num_DA_blind > 0:
        p_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue')
        params_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue',
                       label='Start BE')
    if num_SE_only > 0:
        p_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet')
        params_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet',
                       label='Start PE')
    if filter_ens.est_p:
        max_p, min_p = -np.infty, np.infty
        superscript = '^\mathrm{init}$'
        reference_p = filter_ens.alpha0

        for p, c in zip(filter_ens.est_p, cs):
            mean_p = hist_mean[:, ii].squeeze() / reference_p[p]
            std = np.std(hist[:, ii] / reference_p[p], axis=1)

            max_p, min_p = max(max_p, max(mean_p)), min(min_p, min(mean_p))
            if p in ['C1', 'C2']:
                params_ax.plot(hist_t, mean_p, color=c, label='$' + p + '/' + p + superscript)
            else:
                params_ax.plot(hist_t, mean_p, color=c, label='$\\' + p + '/\\' + p + superscript)

            params_ax.set(xlabel='$t$')
            params_ax.fill_between(hist_t, mean_p + std, mean_p - std, alpha=0.2, color=c)
            ii += 1
        params_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=3)
        params_ax.set(ylim=[min_p - filter_ens.t_CR, max_p + filter_ens.t_CR], xlim=x_lims)

    # PLOT RMS ERROR-------------------------------------------------------------------
    Psi = (hist_mean - hist)[:-Nt_extra]
    Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.) for ti in range(len(Psi))]
    RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]
    RMS_ax.plot(hist_t[:-Nt_extra], RMS, color='firebrick')
    RMS_ax.set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')

    # PLOT COST FUNCTION-------------------------------------------------------------------
    J = np.array(filter_ens.hist_J).squeeze()
    J_ax.plot(t_obs, J[:, :-1])
    dJ_ax.plot(t_obs, J[:, -1], color='tab:red')

    dJ_ax.set(ylabel='$d\\mathcal{J}/d\\psi$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.set(ylabel='$\\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.legend(['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$',
                 '$\\mathcal{J}_{b}$'], bbox_to_anchor=(0., 1.),
                loc="lower left", ncol=3)

    if filename is not None:
        # plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
        plt.close()
    else:
        plt.show()


# ==================================================================================================================
def post_process_multiple(folder, filename=None, k_max=100.):
    data_file = folder + 'CR_data'
    if not os.path.isfile(data_file):
        get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        out = pickle.load(f)

    xlims = [min(out['ks']) - .5, min(k_max, max(out['ks'])) + .5]
    for Li in range(len(out['Ls'])):
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        fig.suptitle(folder)
        subfigs = fig.subfigures(2, 1)
        axCRP = subfigs[0].subplots(1, 3)
        mean_ax = subfigs[1].subplots(1, 2)

        # PLOT CORRELATION AND RMS ERROR  -------------------------------------------------------------------
        ms = 4
        for lbl, col, fill, alph in zip(['biased', 'unbiased'], ['#20b2aae5', '#000080ff'], ['none', 'none'], [.6, .6]):
            for ax_i, key in enumerate(['C', 'R']):
                for suf, mk in zip(['_DA', '_post'], ['o', 'x']):
                    val = out[key + '_' + lbl + suf][Li]
                    axCRP[ax_i].plot(out['ks'], val, linestyle='none', marker=mk, color=col,
                                     label=lbl + suf, markersize=ms, alpha=alph, fillstyle=fill)

        # Plor true and pre-DA RMS and correlation---------------------------------
        for suffix, alph, lw in zip(['true', 'pre'], [.2, 1.], [5., 1.]):
            for ii, key in enumerate(['C_', 'R_']):
                val = out[key + suffix]
                axCRP[ii].plot((-10, 100), (val, val), '-', color='k', label=suffix, alpha=alph, linewidth=lw)

        axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$', ylim=[.95 * out['C_pre'], 1.005])
        axCRP[1].set(ylim=[0., 1.5 * out['R_pre']], ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')

        # PLOT MEAN ERRORS --------------------------------------------------------------------------------------
        for mic in [0]:
            norm = colors.Normalize(vmin=0, vmax=min(k_max, max(out['ks'])) * 1.25)
            cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGn_r)
            for ax, lbl in zip(mean_ax, ['biased', 'unbiased']):
                for ki, kval in enumerate(out['ks']):
                    if kval <= k_max:
                        ax.plot(out['t_interp'], out['error_' + lbl][Li, ki, :, mic] * 100,
                                color=cmap.to_rgba(kval), lw=.9)
                ax.set(xlabel='$t$ [s]', xlim=[out['t_interp'][0], out['t_interp'][-1]])

            mean_ax[0].set(ylim=[0, 60], ylabel='Biased signal error [\%]')
            mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\%]')

        clb = fig.colorbar(cmap, ax=mean_ax[1], orientation='vertical', fraction=0.1)
        clb.ax.set_title('$\\gamma$')

        clb.set_ticks(np.linspace(min(out['ks']), min(k_max, max(out['ks'])), 5))

        # PLOT PARAMETERS AND MEAN EVOLUTION -------------------------------------------------------------------
        flag = True
        for file_k in os.listdir(out['L_dirs'][Li]):
            with open(out['L_dirs'][Li] + file_k, 'rb') as f:
                _ = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
            k = filter_ens.bias.k
            if k > k_max:
                continue
            if filter_ens.est_p:
                if flag:
                    N_psi = len(filter_ens.psi0)
                    c = ['g', 'mediumpurple', 'sandybrown', 'r']
                    superscript = '^\mathrm{init}$'
                    reference_p = filter_ens.alpha0
                    i0 = np.argmin(abs(filter_ens.hist_t - truth['t_obs'][0]))  # start of assimilation
                    i1 = np.argmin(abs(filter_ens.hist_t - truth['t_obs'][-1]))  # end of assimilation
                for pj, p in enumerate(filter_ens.est_p):
                    for idx, a, tt, mk in zip([i1, i0], [1., .2],
                                              ['end', 'start'], ['x', '+']):
                        hist_p = filter_ens.hist[idx - 1, N_psi + pj] / reference_p[p]
                        if p in ['C1', 'C2']:
                            lbl = '$' + p + '/' + p + superscript + '$(t_\mathrm{{}})$'.format(tt)
                        else:
                            lbl = '$\\' + p + '/\\' + p + superscript  + '$(t_\mathrm{{}})$'.format(tt)

                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=a, mew=.8,
                                          fmt=mk, color=c[pj], label=lbl, capsize=4, markersize=4, linewidth=.8)
                if flag:
                    axCRP[2].legend()
                    for ax1 in axCRP[1:]:
                        ax1.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
                    flag = False

        for ax1 in axCRP[:]:
            x0, x1 = ax1.get_xlim()
            y0, y1 = ax1.get_ylim()
            ax1.set_aspect((x1 - x0) / (y1 - y0))

        # SAVE PLOT -------------------------------------------------------------------
        if filename is not None:
            # plt.savefig(filename + '.svg', dpi=350)
            plt.savefig(filename + '_L{}.pdf'.format(out['Ls'][Li]), dpi=350)
            plt.close()


# ==================================================================================================================

def plot_Lk_contours(folder, filename='contour'):
    data_file = folder + 'CR_data'
    # if not os.path.isfile(data_file):
    get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    data = data.copy()
    # -------------------------------------------------------------------------------------------------- #
    R_metrics = [data['R_biased_DA'], data['R_unbiased_DA'], data['R_biased_post'], data['R_unbiased_post']]
    min_idx = [np.argmin(metric) for metric in R_metrics]
    all_datas = [metric.flatten() for metric in R_metrics]
    R_text = ['({:.4},{:.4})'.format(all_datas[0][min_idx[0]], all_datas[1][min_idx[0]]),
              '({:.4},{:.4})'.format(all_datas[0][min_idx[1]], all_datas[1][min_idx[1]]),
              '({:.4},{:.4})'.format(all_datas[2][min_idx[2]], all_datas[3][min_idx[2]]),
              '({:.4},{:.4})'.format(all_datas[2][min_idx[3]], all_datas[3][min_idx[3]])]
    R_metrics = [np.log(metric) for metric in R_metrics]  # make them logs
    R_lbls = ['log(RMS_b)', 'log(RMS_u)', 'log(RMS_b)', 'log(RMS_u)']
    R_titles = ['during DA', 'during DA', 'post-DA', 'post-DA']
    # -------------------------------------------------------------------------------------------------- #
    log_metrics = [np.log(data['R_biased_DA'] + data['R_unbiased_DA']),
                   np.log(data['R_biased_post'] + data['R_unbiased_post'])]
    min_idx = [np.argmin(metric) for metric in log_metrics]
    log_text = ['({:.4},{:.4})'.format(all_datas[0][min_idx[0]], all_datas[1][min_idx[0]]),
                '({:.4},{:.4})'.format(all_datas[2][min_idx[1]], all_datas[3][min_idx[1]])]
    log_lbls = ['log(RMS_b + RMS_u)', 'log(RMS_b + RMS_u)']
    log_titles = ['during DA', 'post-DA']
    # -------------------------------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    fig.suptitle('true R_b = {:.4}, preDA R_b = {:.4}'.format(np.min(data['R_true']), (np.min(data['R_pre']))))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])
    subfigs[0].set_facecolor('0.85')
    # Original and interpolated axes
    xo, yo = data['ks'], data['Ls']
    xm = np.linspace(min(data['ks']), max((data['ks'])), 100)
    ym = np.linspace(min(data['Ls']), max((data['Ls'])), 100)
    # Select colormap
    cmap = mpl.cm.Purples.copy()
    ax0 = None
    for sfig, metrics, lbls, ttls, txts in zip(subfigs, [R_metrics, log_metrics], [R_lbls, log_lbls],
                                               [R_titles, log_titles], [R_text, log_text]):
        max_v = min(0.1, max([np.max(metric) for metric in metrics]))
        min_v = min([np.min(metric) for metric in metrics])
        if lbls[0] == log_lbls[0]:
            min_v, max_v = -1.6, 0.1

        norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
        mylevs = np.linspace(min_v, max_v, 15)
        # Create subplots ----------------------------
        axs = sfig.subplots(2, int(len(metrics) / 2))
        for ax, metric, titl, lbl, txt in zip(axs.flatten(), metrics, ttls, lbls, txts):
            func = interp2d(xo, yo, metric, kind='linear')
            zm = func(xm, ym)
            im = ax.contourf(xm, ym, zm, levels=mylevs, cmap=cmap, norm=norm, extend='max')
            if ax0 is None:
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()

            ax.set_aspect((x1 - x0) / (y1 - y0))

            im.cmap.set_over('k')
            sfig.colorbar(im, ax=ax, label=lbl, shrink=0.6)
            ax.set(title=titl, xlabel='$\\gamma$', ylabel='$L$')

            # find minimum point
            idx = np.argmin(metric)
            idx_j = idx // int(len(data['ks']))
            idx_i = idx % int(len(data['ks']))
            ki, Lj = data['ks'][idx_i], data['Ls'][idx_j]
            ax.plot(ki, Lj, 'ro', markersize=3)
            ax.annotate('({}, {}), {:.4}\n {}'.format(ki, Lj, np.min(metric), txt),
                        xy=(ki, Lj), xytext=(0., 0.), bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 5})
    plt.savefig(filename + '.pdf', dpi=350)
    plt.close()

    # ================================================================================================================
    # minimum at post-DA is the file of interest. Plot best solution timeseties
    k_file = '{}L{}/{}'.format(folder, int(data['Ls'][idx_j]), data['k_files'][idx_i])
    with open(k_file, 'rb') as f:
        params = pickle.load(f)
        truth = pickle.load(f)
        filter_ens = pickle.load(f)
    post_process_single(filter_ens, truth, params, filename=filename + '_optimal_solution_J')


XDG_RUNTIME_DIR = 'tmp/'
if __name__ == '__main__':
    # autumcrisp = 'C:/Users/an553/OneDrive - University of Cambridge/PhD/My_papers/2022.11 - Model-error inference/'
    # myfolder = 'results/VdP_final_.3/'
    # figs_folder = myfolder + 'figs/'

    # fff = '/home/an553/Documents/PycharmProjects/Bias-EnKF/results/Rijke_test/m50_ESN200_L10/'
    #
    # filename = '{}results_CR'.format(fff+'figs/')
    # post_process_multiple(fff, filename)
    #
    # for m in [10, 30, 50, 70, 100]:
    #     myfolder = 'results/Rijke_test/m{}/'.format(m)

    myfolder = 'results/VdP_final_.3/'
    loop_folder = myfolder + 'results_loopParams/'

    # if not os.path.isdir(loop_folder):
    #     continue

    my_dirs = os.listdir(loop_folder)
    for std_item in my_dirs:
        if not os.path.isdir(loop_folder + std_item) or std_item[:3] != 'std':
            continue
        print(loop_folder + std_item)
        std_folder = loop_folder + std_item + '/'

        file = '{}Contour_std{}_results'.format(loop_folder, std_item.split('std')[-1])
        plot_Lk_contours(std_folder, file)

        file = '{}CR_std{}_results'.format(loop_folder, std_item.split('std')[-1])
        post_process_multiple(std_folder, file)
