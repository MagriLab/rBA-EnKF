import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation
from Util import interpolate, CR, getEnvelope
from scipy.interpolate import CubicSpline, interp1d, interp2d
import numpy as np
import matplotlib as mpl
from run import get_error_metrics, compute_CR
from matplotlib.ticker import FormatStrFormatter

XDG_RUNTIME_DIR = 'tmp/'

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=12, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')

# mpl.rcParams('text', usetex=True)
# mpl.rcParams['text.latex.unicode']=True
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

# Figures colors
color_true = 'lightgray'
color_unbias = '#000080ff'
color_bias = '#20b2aae5'
color_obs = 'r'
color_b = 'darkorchid'
colors_alpha = ['green', 'sandybrown', '#8375ff']


def post_process_loopParams(results_dir, figs_dir, k_plot=(None,), k_max=100.):
    if results_dir[-1] != '/':
        results_dir += '/'
    if figs_dir is None:
        figs_dir = results_dir + 'figs/'
    os.makedirs(figs_dir, exist_ok=True)
    std_dirs = os.listdir(results_dir)

    for item in std_dirs:
        if not os.path.isdir(results_dir + item) or item[:3] != 'std':
            continue

        std = item.split('std')[-1]
        filename = '{}Contour_std{}'.format(figs_dir, std)

        plot_Lk(std_dirs, figs_dir, k_plot=k_plot, k_max=k_max, filename=filename)


def plot_Lk(Lk_dir, figs_dir, k_max=100., L_plot=None, k_plot=None, filename=""):
    os.makedirs(figs_dir, exist_ok=True)
    # Plot contours
    name = '{}Contours_{}'.format(figs_dir, filename)
    L_opt, k_opt = plot_Lk_contours(Lk_dir, filename=name)
    optimal_filename = None
    if L_plot is None:
        L_plot = [L_opt]
    else:
        L_plot = np.append(L_opt, L_plot)
    if k_plot is None:
        k_plot = [k_opt]
    else:
        k_plot = np.append(k_opt, k_plot)

    # Plot  error metrics
    name = '{}CR'.format(figs_dir)
    post_process_multiple(Lk_dir, filename=name, k_max=k_max, L_plot=L_plot)

    # Plot optimal and requested timeseries
    L_dirs = os.listdir(Lk_dir)
    for L_item in L_dirs:
        L_folder = Lk_dir + L_item + '/'
        if not os.path.isdir(L_folder) or L_item[0] != 'L':
            continue
        L = L_item.split('L')[-1]
        # Plot error metrics
        if int(L) in L_plot:
            # Plot timeseries
            for k_item in os.listdir(L_folder):
                kval = float(k_item.split('_k')[-1])
                if kval in k_plot:
                    with open(L_folder + k_item, 'rb') as f:
                        params = pickle.load(f)
                        truth = pickle.load(f)
                        filter_ens = pickle.load(f)
                    name = '{}Timeseries_L{}_k{}'.format(figs_dir, L, kval)
                    post_process_single(filter_ens, truth, params, filename=name)

                    if kval == k_opt and int(L) == L_opt:
                        optimal_filename = L_folder + k_item

    return optimal_filename    # Return optimal L-k file


# ==================================================================================================================
def post_process_noise(results_dir, noise_levels=(None,), noise_colors=None, figs_dir=None, plot_contours=True):
    if results_dir[-1] != '/':
        results_dir += '/'
    os.makedirs(figs_dir, exist_ok=True)

    if noise_colors is not None:
        noise_options = noise_colors
        results_dir += 'noise_'
    else:
        noise_options = noise_levels
        results_dir += 'noise'

    if figs_dir is None:
        figs_dir = results_dir + 'figs/'

    # Plot contours and optimal error metrics for each noise level
    data_files = results_dir + 'Optimal_files'
    if not os.path.isfile(data_files) or plot_contours:
        files = []
        for nl in noise_options:
            noise_folder = results_dir + '{}/'.format(nl)
            figs_folder = figs_dir + '{}/'.format(nl)
            print('Plotting error metrics in ...{} '.format(noise_folder.split('results/')[-1]))
            optimal_filename = plot_Lk(noise_folder, figs_dir=figs_folder)
            print('\t Optimal file: {}'.format(optimal_filename.split('results/')[-1]))
            files.append(optimal_filename)

        with open(data_files, 'wb') as f:
            pickle.dump(files, f)
    else:
        with open(data_files, 'rb') as f:
            files = pickle.load(f)

    R_biased, R_unbiased, R_biased_post, R_unbiased_post, R_true, R_pre = [], [], [], [], [], []

    # Plot histogram results
    for filename in files:
        out = compute_CR(filename)
        R_biased.append(out['R_biased_DA'])
        R_unbiased.append(out['R_unbiased_DA'])
        R_biased_post.append(out['R_biased_post'])
        R_unbiased_post.append(out['R_unbiased_post'])
        R_true.append(out['R_true'])
        R_pre.append(out['R_pre'])

        # Plot noise
        with open(filename, 'rb') as f:
            _ = pickle.load(f)
            truth = pickle.load(f)
        truth = truth.copy()
        fig, ax = plt.subplots(3, 1, sharex='all')
        y_true = truth['y']
        noise = truth['noise']
        if 'multi' in truth['noise_type'].lower():
            y_noise = y_true * (1 + noise)
        else:
            mean_y = np.mean(abs(y_true))
            y_noise = y_true + noise * mean_y

        SNR = np.mean(10 * np.log10(np.mean(y_true**2, axis=0) /
                                    np.mean((y_noise - y_true)**2, axis=0)))

        for axi, yy, lbl, c in zip(ax, [y_true, noise, y_noise],
                                   ['true data', 'noise', 'noisy data'], ['b', 'r', 'g']):
            axi.plot(truth['t'], yy[:, 0], color=c)
            axi.legend([lbl])
        ax[2].set(xlim=[truth['t'][-1] - 0.1, truth['t'][-1]])
        fig.suptitle('{}\n{}'.format(truth['noise_type'], SNR))

        plt.savefig(figs_dir + '{}_{}.svg'.format(truth['noise_type'], SNR), dpi=350)
        plt.close()

    bar_width = 0.1
    bars = [np.arange(len(files)),
            [x + bar_width for x in np.arange(len(files))],
            [x + 2 * bar_width for x in np.arange(len(files))],
            [x + 3 * bar_width for x in np.arange(len(files))]]

    # PLOT!
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), layout="constrained")
    colmap = mpl.colormaps['viridis'](np.linspace(0., 1., len(files) * len(bars)))
    lbl = ['biased', 'unbiased', 'biased_post', 'unbiased_post']
    patterns = [None, None, '///', '///']
    for i0, data in enumerate([R_biased, R_unbiased, R_biased_post, R_unbiased_post]):
        c = colmap[i0::len(bars)]
        ax.bar(bars[i0], data, color=c, width=bar_width, edgecolor='k', label=lbl[i0], hatch=patterns[i0])

    for RR, c, lbl, lw in zip([R_true, R_pre], [color_true, 'k'], ['True', 'pre-DA'], [4, 2]):
        ax.axhline(y=np.max(RR), color=c, linewidth=lw)
        ax.axhline(y=np.min(RR), color=c, linewidth=lw, label=lbl)

    ax.set_xticks([r + bar_width for r in range(len(files))], noise_options)
    ax.set(ylabel='RMS error', ylim=[0, 0.6])
    ax.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

    plt.savefig(figs_dir + 'Noise_historgram.svg', dpi=350)
    plt.close()


def post_process_WhyAugment(results_dir, k_plot=None, J_plot=None, figs_dir=None):
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

    colmap = mpl.colormaps['viridis'](np.linspace(0., 1., len(ks) * 2))

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
            y, t, y_unbiased = y[i0 - N_CR:i1 + N_CR], t[i0 - N_CR:i1 + N_CR], y_unbiased[i0 - N_CR:i1 + N_CR]
            y_mean = np.mean(y, -1)

            if flag and ii == 0:
                i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
                i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                y_truth, t_truth = truth['y'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
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

            if filter_ens.bias.k in J_plot:
                filename = '{}WhyAugment_L{}_augment{}_k{}'.format(figs_dir, L, augment, filter_ens.bias.k)
                # print(filename)
                # post_process_single_SE_Zooms(filter_ens, truth, filename=filename)
                post_process_single(filter_ens, truth, params, filename=filename + '_J')

        flag = False

    # --------------------------------------------------------- #

    labels = []
    for kk in ks:
        labels.append('$\\gamma = {}$, U'.format(kk))
        labels.append('$\\gamma = {}$, B'.format(kk))

    bar_width = 0.1
    bars = [np.arange(len(barData[0]))]
    for _ in range(len(ks) * 2):
        bars.append([x + bar_width for x in bars[-1]])

    fig, ax = plt.subplots(1, 2, figsize=(14, 3), layout="constrained")

    for data, br, c, lb in zip(barData, bars, colmap, labels):
        C = np.array([x[0] for x in data]).T.squeeze()
        R = np.array([x[1] for x in data]).T.squeeze()
        Cstd = np.array([x[2] for x in data]).T.squeeze()
        Rstd = np.array([x[3] for x in data]).T.squeeze()
        ax[0].bar(br, C, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[0].errorbar(br, C, yerr=Cstd, fmt='o', capsize=2., color='k', markersize=2)
        ax[1].bar(br, R, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[1].errorbar(br, R, yerr=Rstd, fmt='o', capsize=2., color='k', markersize=2)

    for axi, cr in zip(ax, [(Ct, Cpre), (Rt, Rpre)]):
        axi.axhline(y=cr[0], color=color_true, linewidth=4, label='Truth')
        axi.axhline(y=cr[1], color='k', linewidth=2, label='Pre-DA')
        axi.set_xticks([r + bar_width for r in range(len(data))], xtags)

    ax[0].set(ylabel='Correlation', ylim=[.85, 1.02])
    ax[1].set(ylabel='RMS error', ylim=[0, Rpre * 1.5])
    axi.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=2)

    plt.savefig(figs_dir + 'WhyAugment.svg', dpi=350)
    plt.close()

# ==================================================================================================================


def post_process_single(filter_ens, truth, params, filename=None, mic=0):
    try:
        t_obs, obs = truth['t_obs'], truth['p_obs']
    except KeyError:
        t_obs, obs = truth['t_obs'], truth['y_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    y_filter, t = filter_ens.getObservableHist(), filter_ens.hist_t
    b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

    y_filter = y_filter[:, mic]
    y_mean = np.mean(y_filter, -1)
    b = b[:, mic]
    obs = obs[:, mic]

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

    y_truth = interpolate(truth['t'], truth['y'][:, mic], t)
    std = np.std(y_filter[:, :], axis=1)

    # %% PLOT time series ------------------------------------------------------------------------------------------

    fig1 = plt.figure(figsize=[9, 5.5], layout="constrained")
    subfigs = fig1.subfigures(2, 1, height_ratios=[1, 1.1])
    ax_zoom = subfigs[0].subplots(2, 2, sharex='col', sharey='row')
    ax_all = subfigs[1].subplots(2, 1, sharex='col')

    y_lims = [np.min(y_truth[:N_CR]) * 1.1,
              np.max(y_truth[:N_CR]) * 1.1]

    x_lims = [[t[0], t[0] + 2 * filter_ens.t_CR],
              [t[-1] - 2 * filter_ens.t_CR, t[-1]],
              [t[0], t[-1]]]

    if filter_ens.bias is not None:
        if filter_ens.bias.name == 'ESN':
            t_wash = filter_ens.bias.washout_t
            wash = filter_ens.bias.washout_obs[:, mic]
        b_obs = y_truth - y_mean
        y_lims_b = [np.min(b_obs[:N_CR]) * 1.1, np.max(b_obs[:N_CR]) * 1.1]

    if filter_ens.est_p:
        hist, hist_t = filter_ens.hist, filter_ens.hist_t
        hist_mean = np.mean(hist, -1, keepdims=True)
        mean_p, std_p, labels_p = [], [], []

        max_p, min_p = -np.infty, np.infty
        superscript = '^0$'
        reference_p = filter_ens.alpha0

        ii = len(filter_ens.psi0) - 1
        for p in filter_ens.est_p:
            ii += 1
            m = hist_mean[:, ii].squeeze() / reference_p[p]
            s = np.std(hist[:, ii] / reference_p[p], axis=1)
            max_p, min_p = max(max_p, max(m + 2 * s)), min(min_p, min(m - 2 * s))
            if p in ['C1', 'C2']:
                labels_p.append('$' + p + '/' + p + superscript)
            else:
                labels_p.append('$\\' + p + '/\\bar{\\' + p + '}' + superscript)
            mean_p.append(m)
            std_p.append(s)

    for axs in [ax_zoom[:, 0], ax_zoom[:, 1]]:
        # Observables-----------------------
        axs[0].plot(t, y_truth, color=color_true, linewidth=5, label='t')
        axs[0].plot(t, y_unbiased, '-', color=color_unbias, linewidth=1., label='u')
        axs[0].plot(t, y_mean, '--', color=color_bias, linewidth=1., alpha=0.9, label='b')
        axs[0].fill_between(t, y_mean + std, y_mean - std, alpha=0.2, color='lightseagreen')
        axs[0].plot(t_obs, obs, '.', color=color_obs, markersize=6, label='o')
        axs[0].set(ylim=y_lims)

        # BIAS-----------------------
        if filter_ens.bias is not None:
            if filter_ens.bias.name == 'ESN':
                axs[0].plot(t_wash, wash, '.', color=color_obs, markersize=6)
            axs[1].plot(t, b_obs, color=color_b, label='O', alpha=0.4, linewidth=3)
            axs[1].plot(t_b, b, color=color_b, label='ESN', linewidth=.8)


    ylbls = [["$p(x_\mathrm{f})$ [Pa]", "$b(x_\mathrm{f})$ [Pa]"], ['', '']]
    for axs in [ax_all]:
        if filter_ens.bias is not None:
            axs[0].plot(t, b_obs, color=color_b, label='O', alpha=0.4, linewidth=3)
            axs[0].plot(t_b, b, color=color_b, label='ESN', linewidth=.8)
            axs[0].set(ylabel=ylbls[0][1], xlim=x_lims[-1], ylim=y_lims_b)
        # PARAMS-----------------------
        if filter_ens.est_p:
            for m, s, c, lbl in zip(mean_p, std_p, colors_alpha, labels_p):
                axs[1].plot(hist_t, m, color=c, label=lbl)
                axs[1].set(xlabel='$t$')
                axs[1].fill_between(hist_t, m + s, m - s, alpha=0.2, color=c)

            axs[1].set(xlabel='$t$ [s]', ylabel="", xlim=x_lims[-1], ylim=[min_p, max_p])
            axs[1].plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
            axs[1].plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
            if num_DA_blind > 0:
                axs[1].plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue')
                axs[1].plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue',
                            label='Start BE')
            if num_SE_only > 0:
                axs[1].plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet')
                axs[1].plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet',
                            label='Start PE')
            # axs.legend(loc='best', orientation='horizontal', ncol=3)
            axs[1].legend(loc='upper left', bbox_to_anchor=(0., 1.), ncol=1)
            axs[1].set(ylabel='params')
            axs[0].legend(loc='upper left', bbox_to_anchor=(1., 1.), ncol=1, fontsize='xx-small')

    # axis labels and limits
    for axs, xl, ylbl in zip([ax_zoom[:, 0], ax_zoom[:, 1]], x_lims, ylbls):
        axs[0].set(ylabel=ylbl[0], xlim=xl)
        axs[1].set(ylabel=ylbl[1], xlim=xl, ylim=y_lims_b, xlabel='$t$ [s]')
        for ax_ in axs:
            ax_.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
            ax_.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window

    for ax_ in ax_zoom[:, 0]:
        ax_.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for ax_ in ax_all:
        ax_.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for ax_ in ax_zoom[:, 1]:
        ax_.legend(loc='upper left', bbox_to_anchor=(1., 1.), ncol=1, fontsize='xx-small')

    # # PLOT RMS ERROR-------------------------------------------------------------------
    # Psi = (hist_mean - hist)[:-Nt_extra]
    # Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.) for ti in range(len(Psi))]
    # RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]
    # RMS_ax.plot(hist_t[:-Nt_extra], RMS, color='firebrick')
    # RMS_ax.set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')
    #
    # # PLOT COST FUNCTION-------------------------------------------------------------------
    # J = np.array(filter_ens.hist_J).squeeze()
    # J_ax.plot(t_obs, J[:, :-1])
    # dJ_ax.plot(t_obs, J[:, -1], color='tab:red')
    #
    # dJ_ax.set(ylabel='$d\\mathcal{J}/d\\psi$', xlabel='$t$', xlim=x_lims, yscale='log')
    # J_ax.set(ylabel='$\\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
    # J_ax.legend(['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$',
    #              '$\\mathcal{J}_{b}$'], bbox_to_anchor=(0., 1.),
    #             loc="lower left", ncol=3)

    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.close()
    else:
        plt.show()


# ==================================================================================================================
def post_process_multiple(folder, filename=None, k_max=100., L_plot=None):
    data_file = folder + 'CR_data'
    if not os.path.isfile(data_file):
        get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        out = pickle.load(f)

    xlims = [min(out['ks']) - .5, min(k_max, max(out['ks'])) + .5]

    if L_plot is not None:
        Li_plot = [np.argmin(abs(out['Ls'] - ll)) for ll in L_plot]
    else:
        Li_plot = range(len(out['Ls']))

    for Li in Li_plot:
        fig = plt.figure(figsize=(15, 5), layout="constrained")

        # subfigs = fig.subfigures(1, 2, width_ratios=[1.5, 1])
        subfigs = fig.subfigures(1, 2, width_ratios=[2.45, 2])
        axCRP = subfigs[0].subplots(1, 3)
        mean_ax = subfigs[1].subplots(1, 2)

        # PLOT CORRELATION AND RMS ERROR  -------------------------------------------------------------------
        ms = 4
        for lbl, col in zip(['biased', 'unbiased'], [color_bias, color_unbias]):
            for ax_i, key in enumerate(['C', 'R']):
                for suf, mk in zip(['_DA', '_post'], ['o', 'x']):
                    val = out[key + '_' + lbl + suf][Li]
                    axCRP[ax_i].plot(out['ks'], val, linestyle='none', marker=mk, color=col,
                                     label=lbl[0] + suf, markersize=ms, alpha=.6, fillstyle='none')

        # Plor true and pre-DA RMS and correlation---------------------------------
        for suffix, alph, lw in zip(['true', 'pre'], [.2, 1.], [5., 1.]):
            for ii, key in enumerate(['C_', 'R_']):
                val = out[key + suffix]
                axCRP[ii].plot((-10, 100), (val, val), '-', color='k', label=suffix, alpha=alph, linewidth=lw)

        axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$', ylim=[.95 * out['C_pre'], 1.005])
        axCRP[1].set(ylim=[0., 1.5 * out['R_pre']], ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')
        axCRP[2].set(xlim=xlims, xlabel='$\\gamma$')

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

            mean_ax[0].set(ylim=[0, 60], ylabel='Biased signal error [\\%]')
            mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\\%]')
            mean_ax[1].set_yticks([0., 3., 6., 9.])

        clb = fig.colorbar(cmap, ax=mean_ax.ravel().tolist(), orientation='horizontal', shrink=0.5)
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
                    lbl0, lbl1 = [], []
                    ii = len(filter_ens.psi0) - 1
                    for p in filter_ens.est_p:
                        ii += 1
                        if p in ['C1', 'C2']:
                            lbl0.append('$' + p)
                            lbl1.append('/\\bar{' + p + '}' + '^0$')
                        else:
                            lbl0.append('$\\' + p)
                            lbl1.append('/\\bar{\\' + p + '}' + '^0$')

                for pj, p in enumerate(filter_ens.est_p):
                    hist_p = filter_ens.hist[-1, N_psi + pj] / filter_ens.alpha0[p]
                    lbl = lbl0[pj] + '(t_\mathrm{end})' + lbl1[pj]
                    axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=1, mew=.8, fmt='x',
                                      color=colors_alpha[pj], label=lbl, capsize=4, markersize=4, linewidth=.8)
                if flag:
                    axCRP[2].legend()
                    for ax1 in axCRP[1:]:
                        ax1.legend(loc='best', bbox_to_anchor=(0., 1., 1., 1.), ncol=2, fontsize='xx-small')
                    flag = False

        for ax1 in axCRP[:]:
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.set_aspect(.8 / ax1.get_data_ratio())

        t_obs = truth['t_obs']
        for ax1 in mean_ax[:]:
            ax1.set_aspect(0.6 / ax1.get_data_ratio())
            ax1.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8, alpha=.5)  # DA window
            ax1.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8, alpha=.5)  # DA window

        # SAVE PLOT -------------------------------------------------------------------
        if filename is not None:
            plt.savefig(filename + '_L{}.svg'.format(out['Ls'][Li]), dpi=350)
            plt.close()


# ==================================================================================================================

def plot_Lk_contours(folder, filename='contour'):
    data_file = folder + 'CR_data'
    if not os.path.isfile(data_file):
        get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    data = data.copy()
    # -------------------------------------------------------------------------------------------------- #
    R_metrics = [data['R_biased_DA'], data['R_unbiased_DA'],
                 data['R_biased_post'], data['R_unbiased_post']]
    min_idx = [np.argmin(metric) for metric in R_metrics]
    all_datas = [metric.flatten() for metric in R_metrics]
    R_text = ['({:.4},{:.4})'.format(all_datas[0][min_idx[0]], all_datas[1][min_idx[0]]),
              '({:.4},{:.4})'.format(all_datas[0][min_idx[1]], all_datas[1][min_idx[1]]),
              '({:.4},{:.4})'.format(all_datas[2][min_idx[2]], all_datas[3][min_idx[2]]),
              '({:.4},{:.4})'.format(all_datas[2][min_idx[3]], all_datas[3][min_idx[3]])]
    # R_metrics = [np.log(metric) for metric in R_metrics]  # make them logs

    R_lbls = ['log(RMS_b)', 'log(RMS_u)', 'log(RMS_b)', 'log(RMS_u)']
    R_titles = ['during DA', 'during DA', 'post-DA', 'post-DA']
    # -------------------------------------------------------------------------------------------------- #
    log_metrics = [(data['R_biased_DA'] + data['R_unbiased_DA']),
                   (data['R_biased_post'] + data['R_unbiased_post'])]

    min_idx = [np.argmin(metric) for metric in log_metrics]
    log_text = ['({:.4},{:.4})'.format(all_datas[0][min_idx[0]], all_datas[1][min_idx[0]]),
                '({:.4},{:.4})'.format(all_datas[2][min_idx[1]], all_datas[3][min_idx[1]])]
    log_lbls = ['log(RMS_b + RMS_u)', 'log(RMS_b + RMS_u)']
    log_titles = ['during DA', 'post-DA']
    # -------------------------------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    fig.suptitle('true R_b = {:.4}, preDA R_b = {:.4}'.format(np.min(data['R_true']), (np.min(data['R_pre']))))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])
    subfigs[0].set_facecolor('0.85')

    axs_0 = subfigs[0].subplots(2, 2)
    axs_1 = subfigs[1].subplots(2, 1)

    # Original and interpolated axes
    xo, yo = data['ks'], data['Ls']
    xm = np.linspace(min(data['ks']), max((data['ks'])), 100)
    ym = np.linspace(min(data['Ls']), max((data['Ls'])), 100)

    # Select colormap
    cmap = mpl.cm.RdBu_r.copy()

    min_v = data['R_true']
    mean_v = 1.0
    max_v = max(1.2, data['R_pre'] * 1.2)
    max_cut = data['R_pre']

    L_opt, k_opt = None, None

    for axs, metrics, lbls, ttls, txts in zip([axs_0, axs_1], [R_metrics, log_metrics], [R_lbls, log_lbls],
                                              [R_titles, log_titles], [R_text, log_text]):
        # norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
        norm = mpl.colors.TwoSlopeNorm(vmin=min_v, vcenter=mean_v, vmax=max_v)
        # norm = mpl.colors.TwoSlopeNorm(vmin=np.exp(min_v), vcenter=0.5, vmax=np.exp(max_v))
        mylevs = np.linspace(data['R_true'], max_cut, 9)

        lev_hatches = np.array([0, mylevs[6], (max([np.max(metric) for metric in metrics]) + 1)* mylevs[6]])
        hatches = [None, '////']
        plt.rcParams['hatch.linewidth'] = 0.5

        # Create subplots ----------------------------
        for ax, metric, titl, lbl, txt in zip(axs.flatten(), metrics, ttls, lbls, txts):
            func = interp2d(xo, yo, metric, kind='linear')
            zm = func(xm, ym)
            # im = ax.contourf(xm, ym, zm, cmap=cmap, norm=norm, extend='both', locator=10)
            im = ax.contourf(xm, ym, zm, levels=mylevs, cmap=cmap, norm=norm, extend='both')
            cs = ax.contourf(xm, ym, zm, levels=lev_hatches, colors='None',
                             hatches=hatches)

            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor('silver')
                collection.set_linewidth(0.)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_aspect(1. / ax.get_data_ratio())

            # im.cmap.set_over('k')
            ax.set(title=titl, xlabel='$\\gamma$', ylabel='$L$')

            # find minimum point
            idx = np.argmin(metric)
            idx_j = idx // int(len(data['ks']))
            idx_i = idx % int(len(data['ks']))
            k_opt, L_opt = data['ks'][idx_i], data['Ls'][idx_j]
            ax.plot(k_opt, L_opt, 'w*', markersize=4)
            ax.annotate('({}, {}), {:.4}\n {}'.format(k_opt, L_opt, np.min(metric), txt),
                        xy=(k_opt, L_opt), xytext=(-0.5, -15), bbox={'facecolor': 'white', 'pad': 1}, fontsize=8)

            plt.colorbar(im, ax=ax, label=lbl, shrink=0.7,
                         ticks=mylevs[::2], format=FormatStrFormatter('%.3f'))

    plt.savefig(filename + '.svg', dpi=350)
    plt.close()

    return L_opt, k_opt

    # ================================================================================================================
    # # minimum at post-DA is the file of interest. Plot best solution timeseties
    # k_file = '{}L{}/{}'.format(folder, int(data['Ls'][idx_j]), data['k_files'][idx_i])
    # with open(k_file, 'rb') as f:
    #     params = pickle.load(f)
    #     truth = pickle.load(f)
    #     filter_ens = pickle.load(f)
    #
    # post_process_single(filter_ens, truth, params,
    #                     filename='Timeseries_optimal_L{}_k{}'.format(filter_ens.bias.L, filter_ens.k))
    #
    # filename = filename + '_optimal_solution_CR'
    # post_process_multiple(folder, filename, k_max=20., L_plot=[70])


def plot_Rijke_animation(folder, figs_dir):
    files = os.listdir(folder)

    for ff in files:
        try:
            with open(folder + ff, 'rb') as f:
                params = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
        except:
            continue
        os.makedirs(figs_dir, exist_ok=True)
        filename = '{}results_{}_{}_J'.format(figs_dir, filter_ens.filt, filter_ens.bias.name)

        post_process_single(filter_ens, truth, params, filename=filename, mic=0)

        if filter_ens.filt == 'EnKF':
            filter_ens_BB = filter_ens.copy()
            print('ok1')
        elif filter_ens.filt == 'rBA_EnKF':
            filter_ens_BA = filter_ens.copy()
            print('ok2')

    # load truth
    name_truth = truth['name'].split('_{}'.format(truth['true_params']['manual_bias']))[0]
    with open('data/Truth_{}'.format(name_truth), 'rb') as f:
        truth_ens = pickle.load(f)

    # extract history of observables along the tube
    locs = np.linspace(0, filter_ens.L, 20)
    locs_obs = filter_ens.x_mic

    # Bias-aware EnKF sol
    y_BA = filter_ens_BA.getObservableHist(loc=locs)
    y_BA = np.mean(y_BA, -1)
    # Bias-blind EnKF sol
    y_BB = filter_ens_BB.getObservableHist(loc=locs)
    y_BB = np.mean(y_BB, -1)
    # truth
    y_t = truth_ens.getObservableHist(loc=locs).squeeze()
    y_t += .4 * y_t * np.sin((np.expand_dims(truth_ens.hist_t, -1) * np.pi * 2) ** 2)
    y_t = interpolate(truth_ens.hist_t, y_t, filter_ens.hist_t)

    max_v = [np.max(abs(yy)) for yy in [y_t, y_BA, y_BB]]
    max_v = np.max(max_v)

    # -----------------------

    # fig = plt.figure(figsize=[15, 5], layout='constrained')
    # sub_figs = fig.subfigures(2, 1)
    # ax1 = sub_figs[0].subplots(1, 3, gridspec_kw={'width_ratios': [1, 5, 1]})
    # ax1[0].axis('off')
    # ax1[2].axis('off')
    # ax1 = ax1[1]
    # ax2 = sub_figs[1].subplots(1, 2)

    fig1 = plt.figure(figsize=[10, 2], layout='constrained')
    ax1 = fig1.subplots(1, 1)

    fig2 = plt.figure(figsize=[12, 6], layout='constrained')
    ax2 = fig2.subplots(2, 1)

    t0 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][0] - filter_ens.t_CR / 2)))
    t1 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][-1] + filter_ens.t_CR)))

    t_gif = filter_ens.hist_t[t0:t1:5]

    # all pressure points
    y_BA = filter_ens_BA.getObservableHist(loc=locs)
    y_BB = filter_ens_BB.getObservableHist(loc=locs)
    y_t, y_BB, y_BA = [interpolate(filter_ens.hist_t, yy, t_gif) for yy in [y_t, y_BB, y_BA]]
    max_v = np.max(abs(y_t))

    # observation points
    y_BB_obs = filter_ens_BB.getObservableHist()
    y_BA_obs = filter_ens_BA.getObservableHist()

    y_BA_obs = y_BA_obs[::filter_ens_BA.bias.upsample] + np.expand_dims(filter_ens_BA.bias.hist, -1)

    y_BA_obs = interpolate(filter_ens_BA.bias.hist_t, y_BA_obs, t_gif)
    y_BB_obs = interpolate(filter_ens_BB.hist_t, y_BB_obs, t_gif)

    # parameters
    reference_p = filter_ens_BA.alpha0
    alpha_BA, std_BA, alpha_BB, std_BB = [], [], [], []
    hist_BA, hist_BB = filter_ens_BA.hist, filter_ens_BB.hist
    for pi, p in enumerate(filter_ens.est_p):
        print(pi)
        ii = len(filter_ens.psi0) + pi
        alpha_BA.append(np.mean(hist_BA[:, ii], -1) / reference_p[p])
        std_BA.append(np.std(hist_BA[:, ii] / reference_p[p], axis=1))
        alpha_BB.append(np.mean(hist_BB[:, ii], -1) / reference_p[p])
        std_BB.append(np.std(hist_BB[:, ii] / reference_p[p], axis=1))

    max_p = max([np.max(np.array(a) + np.array(s)) for a, s in zip([alpha_BA, alpha_BB], [std_BA, std_BB])])
    min_p = min([np.min(np.array(a) - np.array(s)) for a, s in zip([alpha_BA, alpha_BB], [std_BA, std_BB])])

    params_legend = []
    for filter_name in ['EnKF', 'BA-EnKF']:
        for p in filter_ens.est_p:
            params_legend.append('$\\' + p + '$ ' + filter_name)

    # timeseries
    y_BA_tt = filter_ens_BA.getObservableHist()[:, 0]
    y_BA_tt_u = y_BA_tt[::filter_ens_BA.bias.upsample] + np.expand_dims(filter_ens_BA.bias.hist[:, 0], -1)

    y_BB_tt = filter_ens_BB.getObservableHist()[:, 0]
    y_t_tt = truth['y'][:, 0]
    y_obs_tt = truth['p_obs'][:, 0]
    pressure_legend = ['Truth', 'Data', 'State + bias  BA', 'State est. BA', 'State est.']

    def animate1(ai):
        ax1.clear()
        ax1.set(ylim=[-max_v, max_v], xlim=[0, 1], title='$t={:.4}$'.format(t_gif[ai]),
                xlabel='$x/L$', ylabel="$p'$ [Pa]")
        ax1.plot(locs, y_t[ai], color=color_true, linewidth=3)
        for loc in filter_ens.x_mic:
            ax1.plot([loc, loc], [0.7 * max_v, max_v], '.-', color='black', linewidth=2)
        ax1.plot([filter_ens.x_mic[0], filter_ens.x_mic[0]], [-max_v, max_v], '--',
                 color='firebrick', linewidth=4, alpha=0.2)
        for yy, c, l in zip([y_BA[ai], y_BB[ai]], [color_bias, 'orange'], ['-', '--']):
            y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
            ax1.plot(locs, y_mean, l, color=c)
            ax1.fill_between(locs, y_mean + y_std, y_mean - y_std, alpha=0.1, color=c)
        # # Plot observables
        # if any(abs(t_gif[ii] - truth['t_obs']) < 1E-6):
        #     jj = np.argmin(abs(t_gif[ai] - truth['t_obs']))
        #     ax1.plot(locs_obs, truth['p_obs'][jj], 'o', color='red', markersize=4,
        #              markerfacecolor=None, markeredgewidth=2)
        #     # for yy, c in zip([y_BA_obs[ai], y_BB_obs[ai]], ['lightseagreen', 'orange']):
        #     #     y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
        #     #     ax1.plot(locs_obs, y_mean, 'x', color=c, markeredgewidth=2)

    def animate2(ai):
        t_g = t_gif[ai]
        # Plot timeseries ------------------------------------------------------------------------
        t11 = np.argmin(abs(filter_ens.hist_t - t_g))
        t00 = np.argmin(abs(filter_ens.hist_t - (t_g - filter_ens.t_CR / 2.)))
        tt_ = filter_ens.hist_t[t00:t11]
        for ax_ in ax2:
            ax_.clear()
            ax_.set(xlim=[tt_[0], tt_[-1] + filter_ens.t_CR * .05], xlabel='$t$ [s]')
        ax2[0].set(ylim=[-max_v, max_v], ylabel="$p'(x/L=0.2)$ [Pa]")
        yy = interpolate(truth['t'], y_t_tt, tt_)
        ax2[0].plot(tt_, yy, color=color_true, linewidth=3)
        ax2[0].plot(truth['t_obs'][0], y_obs_tt[0], 'o', color=color_obs, markersize=3,
                    markerfacecolor=None, markeredgewidth=2)
        yy = interpolate(filter_ens_BA.bias.hist_t, np.mean(y_BA_tt_u, -1), tt_)
        ax2[0].plot(tt_, yy, color=color_unbias, linewidth=1)
        for yy, c, l in zip([y_BA_tt, y_BB_tt], ['lightseagreen', 'orange'], ['-', '--']):
            yy = interpolate(filter_ens.hist_t, yy, tt_)
            y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
            ax2[0].plot(tt_, y_mean, l, color=c)
        ax2[0].legend(pressure_legend, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        for yy, c, l in zip([y_BA_tt, y_BB_tt], [color_bias, 'orange'], ['-', '--']):
            yy = interpolate(filter_ens.hist_t, yy, tt_)
            y_std = np.std(yy, -1)
            ax2[0].fill_between(tt_, y_mean + y_std, y_mean - y_std, alpha=0.1, color=c)
        # # Plot obs data ------------------------------------------------------------------------
        t11_o = np.argmin(abs(truth['t_obs'] - t_g))
        t00_o = np.argmin(abs(truth['t_obs'] - (t_g - filter_ens.t_CR / 2.)))
        ax2[0].plot(truth['t_obs'][t00_o:t11_o], y_obs_tt[t00_o:t11_o], 'o', color=color_obs,
                    markersize=3, markerfacecolor=None, markeredgewidth=2)

        # plot parameters ------------------------------------------------------------------------
        ax2[1].set(ylim=[min_p, max_p], ylabel="")
        for mean_p, std_p, line_type in zip([alpha_BA, alpha_BB], [std_BA, std_BB], ['-', '--']):
            cols = ['mediumpurple', 'orchid']
            for ppi, pp in enumerate(filter_ens.est_p):
                ax2[1].plot(tt_, mean_p[ppi][t00:t11], line_type, color=cols[ppi], label=pp)
        ax2[1].legend(params_legend, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        for mean_p, std_p, line_type in zip([alpha_BB, alpha_BA], [std_BB, std_BA], ['-', '--']):
            cols = ['mediumpurple', 'orchid']
            for ppi, pp in enumerate(filter_ens.est_p):
                ax2[1].fill_between(tt_, mean_p[ppi][t00:t11] + std_p[ppi][t00:t11],
                                    mean_p[ppi][t00:t11] - std_p[ppi][t00:t11], alpha=0.2, color=cols[ppi])

    # Create and save animations ------------------------------------------------------------------------
    ani1 = mpl.animation.FuncAnimation(fig1, animate1, frames=len(t_gif), interval=10, repeat=False)
    ani2 = mpl.animation.FuncAnimation(fig2, animate2, frames=len(t_gif), interval=10, repeat=False)
    writergif = mpl.animation.PillowWriter(fps=10)
    ani1.save(figs_dir + 'ani_tube.gif', writer=writergif)
    ani2.save(figs_dir + 'ani_timeseries.gif', writer=writergif)


if __name__ == '__main__':

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
