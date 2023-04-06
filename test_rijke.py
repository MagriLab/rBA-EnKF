
if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createESNbias, createEnsemble
    from plotResults import *

    bias_form = 'time'
    run_loopParams = 1
    plot_loopParams = 1

    Ls = np.linspace(10, 100, 10, dtype=int)
    if bias_form == 'time':
        ks = np.linspace(0.25, 4.75, 10)
    else:
        ks = np.linspace(0., 10., 41)
    stds = [.25]
    for mm in [50]:

        # %% ========================== SELECT WORKING PATHS ================================= #
        folder = 'results/Rijke_final_{}/'.format(bias_form)
        os.chdir('/mscott/an553')  # set working directory to mscott

        path_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'

        loopParams_folder = folder + 'm{}/results_loopParams_final/'.format(mm)

        if run_loopParams:
            # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
            true_params = {'model': TAModels.Rijke,
                           't_max': 2.5,
                           'beta': 4.2,
                           'tau': 1.4E-3,
                           'manual_bias': bias_form
                           }
            forecast_params = {'model': TAModels.Rijke,
                               't_max': 2.5
                               }
            # ==================================== SELECT FILTER PARAMETERS =================================== #
            filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                             'm': mm,
                             'est_p': ['beta', 'tau'],  #, 'C1', 'C2'],
                             'biasType': Bias.ESN,
                             # Define the observation timewindow
                             't_start': 1.5,  # ensure SS
                             't_stop': 2.0,
                             'kmeas': 20,
                             # Inflation
                             'inflation': 1.002,
                             'start_ensemble_forecast': 1
                             }
            if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
                train_params = {'model': TAModels.Rijke,
                                'std_a': 0.2,
                                'std_psi': 0.2,
                                'est_p': filter_params['est_p'],
                                'alpha_distr': 'uniform',
                                'ensure_mean': True
                                }
                bias_params = {'N_wash': 50,
                               'upsample': 2,
                               'L': 10,
                               'N_units': 500,
                               'augment_data': True,
                               't_val': 0.02,
                               't_train': 0.5,
                               'train_params': train_params,
                               'tikh_': np.array([1e-16]),
                               'sigin_': [np.log10(1e-5), np.log10(1e-2)],
                               }
                if bias_form == 'time':
                    bias_params['t_train'] = 1.5
                    filter_params['kmeas'] = 10
            else:
                bias_params = None
            # ================================== CREATE REFERENCE ENSEMBLE =================================
            name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])
            ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                                   filter_params, bias_params,
                                                   working_dir=folder, filename=name)

            # =========================================== RUN LOOP ==========================================
            for std in stds:
                blank_ens = ensemble.copy()
                # Reset std
                blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                         std, blank_ens.m, method='normal')
                blank_ens.hist[-1] = blank_ens.psi
                blank_ens.std_psi, blank_ens.std_a = std, std
                std_folder = loopParams_folder + 'std{}/'.format(std)
                for L in Ls:
                    # Reset ESN
                    bias_params['L'] = L
                    filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
                    blank_ens.initBias(filter_params['Bdict'])

                    results_folder = std_folder + 'L{}/'.format(L)
                    for k in ks:  # Reset gamma value
                        filter_ens = blank_ens.copy()
                        filter_ens.bias.k = k
                        # Run simulation
                        main(filter_ens, truth, filter_params, results_dir=results_folder, save_=True)
                get_error_metrics(std_folder)

        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        if plot_loopParams:
            if not os.path.isdir(loopParams_folder):
                print('results_loopParams not run')
            else:
                figs_dir = path_dir + loopParams_folder
                post_process_loopParams(loopParams_folder, k_plot=(None,), figs_dir=figs_dir)



