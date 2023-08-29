# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:06:13 2022

@author: an553
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf
import h5py
import os as os

os.environ["OMP_NUM_THREADS"] = '1'  # imposes only one core

import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
from skopt.plots import plot_convergence
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

from scipy.io import savemat
import time

default_params = dict(trainData=None, filename='',
                      dt=1E-4, upsample=5,
                      N_wash=50, t_train=1.0, t_val=0.1,
                      N_units=100, connect=5, noise_level=0.03,
                      test_run=True, augment_data=True,
                      tikh_=np.array([1e-10, 1e-12, 1e-16]),
                      sigin_=[np.log10(1e-4), np.log10(0.5)],
                      rho_=[.7, 1.05],
                      path_dir=''
                      )

inputs = locals().copy()
for key, val in inputs.items():
    if key in default_params.keys():
        default_params[key] = val

locals().update(default_params)


# % LOAD REQUIRED FUNCTIONS AND ENVIRONMENTS
fns_training_file = path_dir + "fns_training.py"
exec(open(fns_training_file).read())

# __________________________________________________

if trainData is None:
    filename = 'data/test_results_bias.npy'
    trainData = np.load(filename)



ESN_filename = '/'.join(filename.split('/')[:-1])
ESN_filename += '/ESN{}_augment{}_L{}.mat'.format(N_units, augment_data, L)

# Force trainData to be (Nalpha, Nt, Nmic)
if len(np.shape(trainData)) == 1:  # (Nt,)
    trainData = np.expand_dims(trainData, 1)
    trainData = np.expand_dims(trainData, 0)
elif len(np.shape(trainData)) == 2:  # (Nt, Nmic)
    trainData = np.expand_dims(trainData, 0)
else:
    if np.argmax(np.shape(trainData)) == 0:
        trainData = trainData.transpose((2, 0, 1))

is_param = trainData.min(axis=1)[0] - trainData.max(axis=1)[0] == 0
if any(is_param):
    alpha = trainData[:, 0, is_param]
    trainData = trainData[:, :, ~is_param]
    norm_alpha = np.mean(0.1 / alpha, axis=0)
else:
    alpha = []

parametrise = False  # THIS WAS DEFINED BEFORE trainData AUGMENTATION, MAY JUST DELETE THE IMPLEMENTAITON
if not parametrise:
    norm_alpha = None

#  ____________________________ APPLY UPSAMPLE - CONVERT INTO ESN dt ______________________________________
dt_ESN = dt * upsample  # ESN time step
trainData = trainData[:, ::upsample]

#  _____________________________________ trainData AUGMENTATION ___________________________________________________
if augment_data:
    l = int(trainData.shape[0])
    U = np.vstack([trainData,
                   trainData[:l] * 1e-1,
                   trainData[-l:] * -1e-2
                   ])
    print('data augment ON')
else:
    U = trainData
    print('data augment OFF')

#  _______________________________ SEPARATE INTO WASH/TRAIN/VAL SETS ________________________________________
N_dim = U.shape[-1]  # dimension of inputs (and outputs)
N_alpha = U.shape[0]  # number of training timeseries
N_train = int(t_train / dt_ESN)
N_val = int(t_val / dt_ESN)  # length of the trainData used is train+val
N_tv = N_train + N_val
N_wtv = N_wash + N_tv  # useful for compact code later

U_trainData = U[:, :N_wtv]
m = np.mean(U_trainData.min(axis=1), axis=0)
M = np.mean(U_trainData.max(axis=1), axis=0)
norm = M - m  # compute norm (normalize inputs by component range)

U_wash = U[:, :N_wash]
U_tv = U[:, N_wash:N_wtv - 1]
Y_tv = U[:, N_wash + 1:N_wtv]

# ___________________________________________ ADD NOISE ______________________________________________________
# Add noise to inputs and targets during training. Larger noise_level promotes stability in long term,
# but hinders time accuracy
U_std = np.std(U, axis=1)
seed = 0
rnd = np.random.RandomState(seed)

for j in range(N_dim):
    for i in range(N_alpha):
        noise = rnd.normal(0, noise_level * U_std[i, j], N_tv - 1)
        U_tv[i, :, j] += noise

# ________________________________________ INITIALISE ESN HYPERPARAMETRES _________________________________________
bias_in = np.array([.1])  # input bias
bias_out = np.array([1.0])  # output bias
sparse = 1 - connect / (N_units - 1)

# _________________________________ GRID SEARCH AND BAYESIAN OPTIMISATION PARAMS ____________________________________
n_tot = 20  # Total Number of Function Evaluations
n_in = 0  # Number of Initial random points

# The first n_grid^2 points are from grid search. If n_grid**2 < n_tot, perform Bayesian Optimization
n_grid = 4
if n_grid > 0:
    x1 = [[None] * 2 for i in range(n_grid ** 2)]
    k = 0
    for i in range(n_grid):
        for j in range(n_grid):
            x1[k] = [rho_[0] + (rho_[1] - rho_[0]) / (n_grid - 1) * i,
                     sigin_[0] + (sigin_[1] - sigin_[0]) / (n_grid - 1) * j]
            k += 1

# range for hyperparameters
search_space = [Real(rho_[0], rho_[1], name='spectral_radius'),
                Real(sigin_[0], sigin_[1], name='input_scaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) * \
          Matern(length_scale=[0.2, 0.2], nu=2.5, length_scale_bounds=(1e-2, 1e1))

print('\n -------------------- HYPERPARAMETER SEARCH ---------------------\n', str(n_grid) + 'x' + str(n_grid) +
      ' grid points and ' + str(n_tot - n_grid ** 2) + ' points with Bayesian Optimization')


# _____________________________________ HYPERPARAMS OPTIMISATION FUN _____________________________________
def g(val):
    # Gaussian Process reconstruction
    b_e = GPR(kernel=kernell,
              normalize_y=True,  # mean = avg of objective fun
              n_restarts_optimizer=3,  # num of random starts
              noise=1e-10,  # for numerical stability
              random_state=10)  # seed
    # Bayesian Optimization
    res = skopt.gp_minimize(val,  # function to minimize
                            search_space,  # bounds
                            base_estimator=b_e,  # GP kernel
                            acq_func="gp_hedge",  # acquisition function
                            n_calls=n_tot,  # num of evaluations
                            x0=x1,  # Initial grid points
                            n_random_starts=n_in,  # num of random inits
                            n_restarts_optimizer=3,  # tries per acquisition
                            random_state=10)  # seed
    return res


# ________________________________________ TRAIN & VALIDATE NETWORK ___________________________________________
ti = time.time()  # check time

val = RVC_Noise  # Which validation strategy
try:
    N_fo = N_fo  # number of folds
except:
    N_fo = 4  # number of folds

N_in = 0  # interval before the first fold
N_fw = (N_train - N_val) // (N_fo - 1)  # num steps forward the validation interval is shifted (evenly spaced)

# Quantities to be saved
tikh_opt = np.zeros(n_tot)  # optimal tikhonov

gps = [None]  # save the gp reconstruction for each network

k, seed = 0, 1
rnd = np.random.RandomState(seed)

# ======================================= Win & W generation ======================================= ##
N_aug = 1
if norm_alpha is not None:
    N_aug += sum(is_param)
Win = lil_matrix((N_units, N_dim + N_aug))
for j in range(N_units):
    Win[j, rnd.randint(0, N_dim + 1)] = rnd.uniform(-1, 1)
Win = Win.tocsr()

W = csr_matrix(rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1 - sparse)))
spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
W = (1 / spectral_radius) * W  # scaled to have unitary spec radius

# ====================================  Bayesian Optimization ====================================== ##
res = g(val)
gp = res.models[-1]
x_iters = np.array(res.x_iters)
f_iters = np.array(res.func_vals)
minimum = np.append(res.x, [tikh_opt[np.argmin(f_iters)], res.fun])
params = gp.kernel_.get_params()
key = sorted(params)
params_gp = np.array([params[key[2]], params[key[5]][0], params[key[5]][1], gp.noise_])

# =========================================  Train Wout =========================================== ##
Wout = train_save_n(U_wash, U_tv, U[:, N_wash + 1:N_wtv], minimum[2], 10 ** minimum[1], minimum[0])
if len(Wout.shape) == 1:
    Wout = np.expand_dims(Wout, axis=1)

print('\n Time per hyperparameter eval.:', (time.time() - ti) / n_tot,
      '\n Best Results: x', minimum[0], 10 ** minimum[1], minimum[2], ', f', -minimum[-1])

pdf = plt_pdf.PdfPages(ESN_filename[:-len('.mat')] + '_Training.pdf')

fig = plt.figure()
plot_convergence(res)
pdf.savefig(fig)
plt.close(fig)

# %%
# ____________________________________ OUTPUTS: PLOT SEARCH AND SAVE trainData ____________________________________
# Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations.
# The GP reconstruction is based on the n_tot function evaluations decided in the search
n_len = 100  # points to evaluate the GP at
xx, yy = np.meshgrid(np.linspace(rho_[0], rho_[1], n_len),
                     np.linspace(sigin_[0], sigin_[1], n_len))
x_x = np.column_stack((xx.flatten(), yy.flatten()))
x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

fig = plt.figure(figsize=[10, 5], tight_layout=True)
# retrieve the gp reconstruction
amin = np.amin([10, f_iters.max()])
# Final GP reconstruction for each realization at the evaluation points
y_pred = np.clip(-gp.predict(x_gp), a_min=-amin,
                 a_max=-f_iters.min()).reshape(n_len, n_len)
# Plot GP Mean
plt.xlabel('Spectral Radius')
plt.ylabel('Input Scaling (log-scale)')
CS = plt.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
cbar = plt.colorbar()
cbar.set_label('-$\log_{10}$(MSE)', labelpad=15)
CSa = plt.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid', alpha=0.3)
#   Plot the n_tot search points
plt.plot(x_iters[:n_grid ** 2, 0], x_iters[:n_grid ** 2, 1], 'v', c='w',
         alpha=0.8, markeredgecolor='k', markersize=10)
plt.plot(x_iters[n_grid ** 2:, 0], x_iters[n_grid ** 2:, 1], 's', c='w',
         alpha=0.8, markeredgecolor='k', markersize=8)

pdf.savefig(fig)
plt.close(fig)

if trainData.shape[0] > 1:
    rows, cols = min(3, int(trainData.shape[0] / 2)), 2
else:
    rows, cols = 1, 1
fig, ax = plt.subplots(3, 2, layout="constrained", figsize=[10, 5])
try:
    ax = ax.flatten()
except:
    ax = [ax]
finally:
    for ii, ax_ in enumerate(ax):
        try:
            ax_.plot(U[ii, N_wash:N_wtv - 1], label='trainData')
            ax_.set(title='l = {}/{}'.format(ii, trainData.shape[0]))
        except:
            break

pdf.savefig(fig)
plt.close(fig)
# ______________________________________________ RUN TEST _____________________________________________________
if test_run:
    # Select number of tests
    N_t0 = N_wtv + 10  # start test after washout, training and validation trainData
    max_test_time = np.shape(U[:, N_t0:])[1] * dt_ESN
    N_test = 50

    if N_test * t_val > max_test_time:
        N_test = int(np.floor(max_test_time / t_val))

    # Break if not enough trainData for testing
    if N_test < 1:
        print('Test not performed. Not enough trainData')
    else:
        medians_alpha = []
        maxs_alpha = []
        alph = None
        sub_loop = 5  # number of closed loop re-initialisations with true trainData
        N_reinit = N_val // sub_loop  # time between closed loop re-initialisations

        # load matrices and hyperparameters
        rho = minimum[0].copy()
        sigma_in = 10 ** minimum[1].copy()
        for kk in range(N_alpha):
            if norm_alpha is not None:
                alph = alpha[kk]
            subplots = min(10, N_test)  # number of plotted intervals
            # plt.figure(figsize=[5, 10], tight_layout=True)
            plt.subplots(subplots, 1, figsize=[10, 2*subplots])
            errors = np.zeros(N_test)
            # Different intervals in the test set
            i, ii = -1, -1
            while True:
                i += 1
                ii += 1
                if i >= N_test:
                    break
                # trainData for washout and target in each interval
                U_wash = U[kk, N_t0 - N_wash + i * N_val: N_t0 + i * N_val]
                Y_t = U[kk, N_t0 + i * N_val: N_t0 + i * N_val + N_reinit * sub_loop]
                # washout for each interval
                xa1 = open_loop(U_wash, np.zeros(N_units), sigma_in, rho, alph)[-1]
                Yh_t, xa1 = closed_loop(N_reinit - 1, xa1, Wout, sigma_in, rho, alph)
                try:
                    # Do the multiple sub_loop inside each test interval
                    if sub_loop > 1:
                        for j in range(sub_loop - 1):
                            Y_start = Y_t[(j + 1) * N_reinit - 1].copy()  #
                            Y1, xa1 = closed_loop(N_reinit, xa1, Wout, sigma_in, rho, alph)
                            Yh_t = np.concatenate((Yh_t, Y1[1:]))
                    errors[i] = np.log10(np.mean((Yh_t - Y_t) ** 2) / np.mean(norm ** 2))
                    if i < subplots:
                        plt.subplot(subplots, 1, i + 1)
                        plt.plot(np.arange(N_reinit * sub_loop) * dt_ESN, Y_t[:, ii], 'k',
                                 label='truth N_dim ' + str(ii))
                        plt.plot(np.arange(N_reinit * sub_loop) * dt_ESN, Yh_t[:, ii], '--r',
                                 label='ESN N_dim ' + str(ii))
                        plt.legend(title='Test \#' + str(i), loc='upper left', bbox_to_anchor=(1.01, 1.01))
                    if ii == N_dim - 1:
                        ii = -1
                except:
                    raise Exception('error at i = ', i, ', N_test = ', N_test)
            maxs_alpha.append(errors.max())
            medians_alpha.append(np.median(errors))
            plt.tight_layout()
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)
        print('Median and max error in', N_alpha, ' test:', np.median(medians_alpha), max(maxs_alpha))

# _________________________________________ Save output and images ___________________________________

save_dict = dict(t_train=t_train,
                 t_val=t_val,
                 norm=norm,
                 Win=Win.T,
                 Wout=Wout,
                 W=W,
                 dt_ESN=dt_ESN,
                 augment_data=augment_data,
                 N_augment=int(N_alpha),
                 N_wash=int(N_wash),
                 N_units=int(N_units),
                 N_dim=int(N_dim),
                 bias_in=bias_in,
                 bias_out=bias_out,
                 rho=minimum[0],
                 sigma_in=10 ** minimum[1],
                 tikh=minimum[2],
                 upsample=int(upsample),
                 hyperparameters=[minimum[0], 10 ** minimum[1], bias_in[0]],
                 training_time=(N_train + N_val) * dt_ESN,
                 filename=filename,
                 connect=connect
                 )
if norm_alpha is not None:
    save_dict['norm_alpha'] = norm_alpha

savemat(ESN_filename, save_dict, oned_as='column')

pdf.close()

print('\n --------------------------------------------------------------- \n')

# %% ====================================================================== %%#
