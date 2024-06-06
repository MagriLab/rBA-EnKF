if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, create_ESN_train_dataset, createEnsemble
    from plotResults import *
    import os as os

    # %% ================================ SELECT PROPERTIES ================================= #
    bias_form = 'periodic'
    run_noise_levels, plot_noise_levels = 1, 1
    run_noise_colors, plot_noise_colors = 1, 1

    # Loop parameters
    Ls = np.linspace(start=10, stop=90, num=5, dtype=int)
    ks = np.linspace(start=0.5, stop=3.5, num=7)
    noise_colors = ('white', 'pink', 'brown')
    noise_levels = (.001, .01, .1, .2, .3)

    # %% ================================ SELECT WORKING PATHS ================================= #
    path_dir = os.path.realpath(__file__).split('main')[0]  # path to figures
    folder = 'results/Rijke_final_{}/m50/'.format(bias_form)

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model':       TAModels.Rijke,
                   't_max':       2.5,
                   'beta':        4.2,
                   'tau':         1.4E-3,
                   'manual_bias': bias_form,
                   'std_obs':     0.01,
                   'noise_type':  'additive'
                   }
    forecast_params = {'model': TAModels.Rijke,
                       't_max': 2.5
                       }
    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filt':                    'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'm':                       50,
                     'est_p':                   ['beta', 'tau'],
                     'biasType':                Bias.ESN,
                     # Define the observation time window
                     't_start':                 1.5,  # ensure SS
                     't_stop':                  2.0,
                     'kmeas':                   20,
                     # Inflation
                     'inflation':               1.002,
                     'start_ensemble_forecast': 1
                     }

    train_params = {'model':       TAModels.Rijke,
                    'std_a':       0.2,
                    'std_psi':     0.2,
                    'est_p':       filter_params['est_p'],
                    'alpha_distr': 'uniform',
                    'ensure_mean': True
                    }

    bias_params = {'N_wash':       50,
                   'upsample':     2,
                   'L':            1,
                   'N_units':      500,
                   'augment_data': True,
                   't_val':        0.02,
                   't_train':      0.5,
                   'train_params': train_params,
                   'tikh_':        np.array([1e-16]),
                   'sigin_':       [np.log10(1e-5), np.log10(1e-2)],
                   }

    # ==================================== MAIN LOOP FUNCTION =================================== #
    def run_Lk_loop(truth_dict, parent_dir, ensemble_dir):
        #  CREATE REFERENCE ENSEMBLE ------------------------------------------------
        ensemble, truth, args = createEnsemble(truth_dict, forecast_params,
                                               filter_params, bias_params,
                                               working_dir=parent_dir,
                                               filename='reference_Ensemble', ensemble_dir=ensemble_dir)
        std = .25
        noise_ens = ensemble.copy()

        #  RESER ENSEMBLE STD ------------------------------------------------------
        noise_ens.psi = noise_ens.addUncertainty(np.mean(noise_ens.psi, axis=1), std, noise_ens.m)
        noise_ens.hist[-1] = noise_ens.psi
        noise_ens.std_psi, noise_ens.std_a = std, std

        for L in Ls:
            results_folder = noise_dir + 'L{}/'.format(L)
            blank_ens = noise_ens.copy()

            # Reset ESN -------------------------------------------------------------
            bias_paramsL = bias_params.copy()
            bias_paramsL['L'] = L
            Bdict = create_ESN_train_dataset(*args, bias_param=bias_paramsL)
            blank_ens.initBias(Bdict)

            for k in ks:  # Reset gamma value
                filter_ens = blank_ens.copy()
                filter_ens.bias.k = k
                # Run simulation------------------------------------------------------------
                main(filter_ens, truth, method='rBA_EnKF', results_dir=results_folder, save_=True)


    # ==================================== RUN SIMULATIONS =================================== #
    if run_noise_levels:
        working_dir = folder + 'results_noise_levels/'
        for noise_level in noise_levels:
            noise_color = 'gauss'
            noise_dir = working_dir + 'noise{}/'.format(noise_level)

            # Run simulations
            true_params['std_obs'] = noise_level
            true_params['noise_type'] = 'gauss, additive'
            run_Lk_loop(true_params, folder, noise_dir)

    if run_noise_colors:
        working_dir = folder + 'results_noise_colors_{}/'.format(true_params['std_obs'])
        for noise_color in noise_colors:
            noise_dir = working_dir + 'noise_{}/'.format(noise_color)
            # Run simulations
            true_params['std_obs'] = 0.5
            true_params['noise_type'] = 'gauss, additive'
            run_Lk_loop(true_params, folder, noise_dir)

    # ==================================== PLOT RESULTS =================================== #

    if plot_noise_levels:
        working_dir = folder + 'results_noise_levels/'
        post_process_noise(working_dir, noise_levels=noise_levels,
                           figs_dir=path_dir + working_dir, plot_contours=False)

    if plot_noise_colors:
        working_dir = folder + 'results_noise_colors_{}/'.format(true_params['std_obs'])
        post_process_noise(working_dir, noise_colors=noise_colors,
                           figs_dir=path_dir + working_dir, plot_contours=False)
