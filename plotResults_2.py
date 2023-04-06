import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
from Util import interpolate, CR, getEnvelope
from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import matplotlib as mpl
#


# ==================================================================================================================
def post_process_multiple_2(folder, filename=None):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


    files = os.listdir(folder)
    flag = True
    biases, esn_errors, biases_ESN = [], [], []
    ks, CBs, RBs, CUs, RUs, Cpres, Rpres = [], [], [], [], [], [], []
    # ==================================================================================================================
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    fig.suptitle(folder)
    subfigs = fig.subfigures(2, 1)
    axCRP = subfigs[0].subplots(1, 3)
    mean_ax = subfigs[1].subplots(1, 2)
    for file in files:
        # if file[-3:] == '.py' or file[-4] == '.':
        #     continue
        if file.find('_k') == -1:
            continue
        k = float(file.split('_k')[-1])
        # if k > 10: # uncomment these lines to avoid ploting values over 20
        #     continue
        with open(folder + file, 'rb') as f:
            parameters = pickle.load(f)
            truth = pickle.load(f)
            filter_ens = pickle.load(f)
        # Observable bias
        y_filter, t_filter = filter_ens.getObservableHist(), filter_ens.hist_t
        y_truth, t_truth = truth['y'], truth['t']
        if flag:
            N_CR = int(filter_ens.t_CR / filter_ens.dt)  # Length of interval to compute correlation and RMS
            N_mean = int(.1 / filter_ens.dt)  # Length of interval to average mean error
            istart = np.argmin(abs(t_filter - truth['t_obs'][0]))  # start of assimilation
            istop = np.argmin(abs(t_filter - truth['t_obs'][parameters['num_DA'] - 1]))  # end of assimilation
            istart_t = np.argmin(abs(t_truth - truth['t_obs'][0]))  # start of assimilation
            istop_t = np.argmin(abs(t_truth - truth['t_obs'][parameters['num_DA'] - 1]))  # end of assimilation
            t_interp = t_filter[N_mean::N_mean]



        # ESN bias
        b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
        b_ESN = interpolate(t_b, b, t_filter)

        # Ubiased signal error

        if filter_ens.bias.name == 'ESN':
            y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            y_unbiased = interpolate(t_b, y_unbiased, t_filter)
        else:
            y_unbiased = y_filter + np.expand_dims(b, -1)



        y_filter, t_filter = y_filter[istart-N_CR:istop+N_CR], t_filter[istart-N_CR:istop+N_CR]
        y_truth, t_truth = y_truth[istart_t-N_CR:istop_t+N_CR], t_truth[istart_t-N_CR:istop_t+N_CR]
        y_unbiased = y_unbiased[istart-N_CR:istop+N_CR]

        b_obs_u = y_truth - np.mean(y_unbiased, -1)

        b_obs = y_truth - np.mean(y_filter, -1)

        # compute mean error, time averaged over a kinterval
        bias, bias_esn, esn_err = [], [], []
        for j in range(int(len(b_obs) // N_mean)):
            i = j * N_mean
            mean_bias_obs = np.mean(abs(b_obs[i:i + N_mean]), 0)
            mean_bias_esn = np.mean(abs(b_ESN[i:i + N_mean]), 0)
            # mean_unbiased_error = np.mean(abs(b_obs[i:i + kinterval] - b_ESN[i:i + kinterval]), 0)
            mean_unbiased_error = np.mean(abs(b_obs_u[i:i + N_mean]), 0)

            bias.append(mean_bias_obs)
            bias_esn.append(mean_bias_esn)
            esn_err.append(mean_unbiased_error)

        biases.append(np.array(bias))
        biases_ESN.append(np.array(bias_esn))
        esn_errors.append(np.array(esn_err))
        ks.append(k)

        # PLOT CORRELATION AND RMS ERROR =====================================================================
        t_obs = truth['t_obs'][:parameters['num_DA']]
        y_obs = interpolate(t_filter, y_truth, t_obs)
        y_obs_b = interpolate(t_filter, np.mean(y_filter, -1), t_obs)
        y_obs_u = interpolate(t_filter, np.mean(y_unbiased, -1), t_obs)

        CB, RB = CR(y_truth[-N_CR:], np.mean(y_filter, -1)[-N_CR:])  # biased
        CU, RU = CR(y_truth[-N_CR:], np.mean(y_unbiased, -1)[-N_CR:])  # unbiased
        # Correlation
        bias_c = 'tab:red'
        unbias_c = 'tab:blue'
        ms = 4

        axCRP[0].plot(k, CB, 'o', color=bias_c, label='Biased', markersize=ms, alpha=0.6)
        axCRP[0].plot(k, CU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')
        # RMS error
        axCRP[1].plot(k, RB, 'o', color=bias_c, label='Biased ', markersize=ms, alpha=0.6)
        axCRP[1].plot(k, RU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')

        CB, RB = CR(y_obs, y_obs_b)  # biased
        CU, RU = CR(y_obs, y_obs_u)  # unbiased
        # Correlation
        bias_c = 'tab:red'
        unbias_c = 'tab:blue'
        axCRP[0].plot(k, CB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)
        # RMS error
        axCRP[1].plot(k, RB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)

        # Parameters ========================================================================================
        if filter_ens.est_p:
            if flag:
                N_psi = len(filter_ens.psi0)
                # c = ['tab:orange', 'navy', 'darkcyan', 'cyan']
                c = ['g', 'mediumpurple', 'sandybrown', 'r']
                # c = ['navy', 'chocolate', 'mediumturquoise', 'lightseagreen', 'cyan']
                marker = ['x', '+']
                time = ['$(t_\mathrm{end})$', '$(t_\mathrm{start})$']
                alphas = [1., .2]
                superscript = '^\mathrm{init}$'
                reference_p = filter_ens.alpha0
            for jj, p in enumerate(filter_ens.est_p):
                for kk, idx in enumerate([istop, istart]):
                    hist_p = filter_ens.hist[idx - 1, N_psi + jj] / reference_p[p]
                    if p in ['C1', 'C2']:
                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=alphas[kk],
                                          fmt=marker[kk], color=c[jj], label='$'+p+'/'+ p + superscript + time[kk],
                                          capsize=ms, markersize=ms)
                    else:
                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=alphas[kk],
                                          fmt=marker[kk], color=c[jj], label='$\\'+p+'/\\' + p + superscript + time[kk],
                                          capsize=ms, markersize=ms)

        if flag:
            # compute and plot the baseline correlation and MSE
            if len(truth['b']) > 1:
                b_truth = truth['b'][:len(y_filter)]
                y_truth_u = y_truth - b_truth
                Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
                axCRP[0].plot((-10, 100), (Ct, Ct), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
                axCRP[1].plot((-10, 100), (Rt, Rt), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)

            # compute C and R before the assimilation (the initial ensemble has some initialisation error)
            Cpre, Rpre = CR(y_truth[istart - N_CR:istart + 1:], np.mean(y_filter, -1)[istart - N_CR:istart + 1:])
            axCRP[0].plot((-10, 100), (Cpre, Cpre), '-', color='k', label='Pre-DA')
            axCRP[1].plot((-10, 100), (Rpre, Rpre), '-', color='k', label='Pre-DA')
            axCRP[1].set(ylim=[0., 2. * Rpre])
            for ax1 in axCRP[1:]:
                ax1.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
            flag = False

    # =========================================================================================================
    xlims = [min(ks) - .5, max(ks) + .5]
    axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$')
    axCRP[1].set(ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')


    for ax1 in axCRP[:]:
        x0, x1 = ax1.get_xlim()
        y0, y1 = ax1.get_ylim()
        ax1.set_aspect((x1 - x0) / (y1 - y0))

    # # PLOT MEAN ERROR EVOLUTION ================================================================================
    # for mic in [0]:
    #     scale = np.max(truth['y'][:, mic])
    #     norm = colors.Normalize(vmin=0, vmax=max(ks))
    #     cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    #     for i, metric in enumerate([biases, esn_errors]):  # , biases_ESN]):
    #         errors = [b[:, mic] / scale for b in metric]
    #         for err, k in zip(errors, ks):
    #             mean_ax[i].plot(t_interp, err * 100, color=cmap.to_rgba(k))
    #         mean_ax[i].set(xlim=[t_filter[istart] - 0.02, t_filter[istop] + 0.05], xlabel='$t$ [s]')
    #
    #     mean_ax[0].set(ylim=[0, 60], ylabel='Biased signal error [\%]')
    #     mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\%]')
    #
    #     for i in range(2):
    #         x0, x1 = mean_ax[i].get_xlim()
    #         y0, y1 = mean_ax[i].get_ylim()
    #         # print( (x1 - x0) / (y1 - y0))
    #         mean_ax[i].set_aspect(0.5 * (x1 - x0) / (y1 - y0))
    #
    # clb = fig.colorbar(cmap, ax=mean_ax[1], orientation='vertical', fraction=0.1)
    # clb.ax.set_title('$\\gamma$')

    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
    else:
        plt.show()