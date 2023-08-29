
if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, create_ESN_train_dataset, createEnsemble
    from plotResults import *
    import os as os

    bias_form = 'periodic'
    run_default_kL, run_find_optimal_kL, plot_noise = 0, 1, 1

    # Loop parameters
    Ls = np.linspace(start=10, stop=90, num=5, dtype=int)
    ks = np.linspace(start=0.5, stop=3.5, num=7)
    noise_colors = ('white', 'pink', 'brown')
    noise_levels = (.001, .01, .1, .2, .3)

    noise_colors = ['brown']
    Ls = [70]
    ks = ks[1:]


    # %% ================================ SELECT WORKING PATHS ================================= #
    folder = 'results/Rijke_final_{}/'.format(bias_form)
    path_dir = os.path.realpath(__file__).split('main')[0]
    os.chdir('/mscott/an553/')  # set working directory to mscott

    if run_find_optimal_kL or run_default_kL:
        for noise_color in noise_colors:
            noise_level = 0.25
            working_dir = folder + 'm50/results_noise_colors_strong/'
            noise_dir = working_dir + 'noise_{}/'.format(noise_color)

        # for noise_level in noise_levels:
        #     noise_color = 'gauss'
        #     working_dir = folder + 'm50/results_noise_08_24/'
        #     noise_dir = working_dir + 'noise{}/'.format(noise_level)

            # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
            true_params = {'model': TAModels.Rijke,
                           't_max': 2.5,
                           'beta': 4.2,
                           'tau': 1.4E-3,
                           'manual_bias': bias_form,
                           'std_obs': noise_level,
                           'noise_type': noise_color + ' additive'
                           }
            forecast_params = {'model': TAModels.Rijke,
                               't_max': 2.5
                               }
            # ==================================== SELECT FILTER PARAMETERS =================================== #
            filter_params = {'filt': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                             'm': 50,
                             'est_p': ['beta', 'tau'],
                             'biasType': Bias.ESN,
                             # Define the observation time window
                             't_start': 1.5,  # ensure SS
                             't_stop': 2.0,
                             'kmeas': 20,
                             # Inflation
                             'inflation': 1.002,
                             'start_ensemble_forecast': 1
                             }

            train_params = {'model': TAModels.Rijke,
                            'std_a': 0.2,
                            'std_psi': 0.2,
                            'est_p': filter_params['est_p'],
                            'alpha_distr': 'uniform',
                            'ensure_mean': True
                            }

            bias_params = {'N_wash': 50,
                           'upsample': 2,
                           'L': 1,
                           'N_units': 500,
                           'augment_data': True,
                           't_val': 0.02,
                           't_train': 0.5,
                           'train_params': train_params,
                           'tikh_': np.array([1e-16]),
                           'sigin_': [np.log10(1e-5), np.log10(1e-2)],
                           }

            # ================================== CREATE REFERENCE ENSEMBLE ======================================
            name = 'reference_Ensemble'
            ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                                   filter_params, bias_params,
                                                   working_dir=working_dir,
                                                   filename=name, ensemble_dir=noise_dir)
            std = .25

            # ---------------------------------------------------------------------------------------------------
            if run_default_kL:
                # Part I: compare noise levels with fixed L-gamma ==========================
                filter_ens = ensemble.copy()
                L, k = 60, 2.75

                # Reset std
                filter_ens.psi = filter_ens.addUncertainty(np.mean(filter_ens.psi, axis=1), std, filter_ens.m)
                filter_ens.hist[-1] = filter_ens.psi
                filter_ens.std_psi, filter_ens.std_a = std, std

                # Reset bias L and ESN
                bias_params['L'] = L
                Bdict = create_ESN_train_dataset(*args, bias_param=bias_params)
                filter_ens.initBias(Bdict)
                filter_ens.bias.k = k

                # run simulation
                main(filter_ens, truth, method='rBA_EnKF', results_dir=noise_dir, save_=True)

            if run_find_optimal_kL:
                # Part II: loop over parameters to find optimal L-gamma ==========================
                noise_ens = ensemble.copy()

                # Reset std
                noise_ens.psi = noise_ens.addUncertainty(np.mean(noise_ens.psi, axis=1), std, noise_ens.m)
                noise_ens.hist[-1] = noise_ens.psi
                noise_ens.std_psi, noise_ens.std_a = std, std

                for L in Ls:
                    results_folder = noise_dir + 'L{}/'.format(L)
                    blank_ens = noise_ens.copy()

                    # Reset ESN
                    bias_paramsL = bias_params.copy()
                    bias_paramsL['L'] = L
                    Bdict = create_ESN_train_dataset(*args, bias_param=bias_paramsL)
                    blank_ens.initBias(Bdict)

                    for k in ks:  # Reset gamma value
                        filter_ens = blank_ens.copy()
                        filter_ens.bias.k = k
                        # Run simulation
                        main(filter_ens, truth, method='rBA_EnKF', results_dir=results_folder, save_=True)

                get_error_metrics(noise_dir)

    if plot_noise:

        # noise_folder = folder + 'm50/results_noise_08_24/'
        # post_process_noise(noise_folder, noise_levels=noise_levels,
        #                    figs_dir=path_dir + noise_folder, plot_contours=True)

        noise_folder = folder + 'm50/results_noise_colors_strong/'
        noise_colors = ('white', 'pink', 'brown')
        post_process_noise(noise_folder, noise_colors=noise_colors,
                           figs_dir=path_dir + noise_folder, plot_contours=False)
