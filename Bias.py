import os
# os.environ["OMP_NUM_THREADS"]= '1'
import numpy as np
import scipy.linalg as la
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d


class Bias:
    def __init__(self, b, t):
        self.b = b
        self.t = t
        self.hist = None
        self.hist_t = None
        self.N_dim = len(b)

    def updateHistory(self, b, t, reset=False):
        if self.hist is not None and not reset:
            self.hist = np.concatenate((self.hist, b))
            self.hist_t = np.concatenate((self.hist_t, t))
        else:
            self.hist = np.array([self.b])
            self.hist_t = np.array([self.t])
        self.b = self.hist[-1]
        self.t = self.hist_t[-1]

    def getOutputs(self):
        out = dict(name=self.name,
                   hist=self.hist,
                   hist_t=self.hist_t)
        for key in self.attrs.keys():
            out[key] = getattr(self, key)
        return out

    def updateCurrentState(self, b, t):
        self.b = b
        self.t = t


# =================================================================================================================== #

class NoBias(Bias):
    name = 'None'
    attrs = {}

    def __init__(self, y, t, Bdict=None):
        super().__init__(np.zeros(len(y)), t)

    def getBias(self, *args):
        return self.b

    def stateDerivative(self, y):
        return np.zeros([len(self.b), len(self.b)])

    def timeIntegrate(self, t, y=None, t_end=0):
        return np.zeros([len(t), len(self.b)]), t


# =================================================================================================================== #

class ESN(Bias):
    name = 'ESN'
    attrs = {'t_train': 1.0,
             't_val': 0.1,
             'N_wash': 50,
             'N_units': 100,
             'upsample': 5,
             'test_run': True,
             'L': 1,
             'k': 0.,
             'augment_data': True,
             'washout_obs': None,
             'washout_t': None,
             }

    def __init__(self, y, t, Bdict=None):
        if Bdict is None:
            Bdict = {'folder': 'data/'}
        else:
            for key, val in self.attrs.items():
                if key not in Bdict.keys():
                    setattr(self, key, val)
                    Bdict[key] = val
                else:
                    setattr(self, key, Bdict[key])
        # ------------------------ Define bias data filename ------------------------ #
        self.trainESN(Bdict)
        # -----------  Initialise reservoir state and its history to zeros ------------ #
        self.r = np.zeros(self.N_units)
        self.hist_r = np.array([self.r])
        self.initialised = False
        # --------------------------  Initialise parent Bias  ------------------------- #
        b = np.zeros(self.N_dim)

        super().__init__(b, t)

    def trainESN(self, Bdict):
        # --------------------- Train a new ESN if not in folder --------------------- #
        # ESN_filename = Bdict['filename'][:-len('bias')] + \
        #                'ESN{}_augment{}'.format(self.N_units, self.augment_data)

        ESN_filename = '/'.join(Bdict['filename'][:-len('bias')] .split('/')[:-1])
        ESN_filename += '/ESN{}_augment{}_L{}'.format(self.N_units, self.augment_data, self.L)

        # Check that the saved ESN has the same parameters as the wanted one
        flag = False
        if os.path.isfile(ESN_filename + '.mat'):
            fileESN = loadmat(ESN_filename)
            for key, val in fileESN.items():
                if key in Bdict.keys():
                    if key == 'filename':
                        continue
                    if any(val != Bdict[key]):
                        flag = True
                        print('\n Retraining ESN as {} = {} != {}'.format(key, val, Bdict[key]))
                        break

        if not os.path.isfile(ESN_filename + '.mat') or flag:
            # Load or create bias data
            if 'trainData' in Bdict.keys():
                # print('\t_interp saving bias data')
                bias = Bdict['trainData']
                np.savez(Bdict['filename'], bias)
            else:
                Bdict['trainData'] = np.load(Bdict['filename'] + '.npz')['bias']

            self.N_dim = Bdict['trainData'].shape[1]
            # Delete unnecessary data. Keep only wash + training + val (+ test)
            N_wtv = int((self.t_train + self.t_val) / Bdict['dt']) + self.N_wash * self.upsample
            N_wtv += int(self.t_val * min(10, self.N_dim + 1) / Bdict['dt'])

            if N_wtv > len(bias):
                raise ValueError('Not enough data for training. Increase t_max')

            Bdict['trainData'] = Bdict['trainData'][-N_wtv:]
            # Run training main script
            path_dir = os.path.realpath(__file__).split(__name__+'.py')[0]
            main_training_file = path_dir + "main_training.py"
            Bdict['path_dir'] = path_dir
            exec(open(main_training_file).read(), Bdict)

        # --------------------------- Load trained ESN --------------------------- #
        fileESN = loadmat(ESN_filename)
        for key, val in fileESN.items():
            if key[0] != '_':
                try:
                    if key in ['N_wash', 'N_units', 'N_dim', 'upsample', 'N_augment']:
                        setattr(self, key, int(val))
                    # elif np.shape(val)[-1] == 1:
                    elif key in ['dt_ESN', 'rho', 'sigma_in', 'upsample']:
                        setattr(self, key, float(val))
                    elif key == 'augment_data':
                        setattr(self, key, bool(val))
                    else:
                        setattr(self, key, np.squeeze(val, axis=1))
                except:
                    setattr(self, key, val)

        # --------------------- Create washout observed data ---------------------- #
        # self.N_wash = 1
        self.washout_obs = np.flip(Bdict['washout_obs'][:-self.N_wash * self.upsample:-self.upsample], axis=0)
        if len(self.washout_obs.shape) > 2:
            self.washout_obs = self.washout_obs.squeeze()
        self.washout_t = np.flip(Bdict['washout_t'][:-self.N_wash * self.upsample:-self.upsample])

        assert self.washout_t[-1] == Bdict['washout_t'][-1]
        assert len(self.washout_t) == self.N_wash

        if len(self.Wout.shape) == 1:
            self.Wout = np.expand_dims(self.Wout, axis=1)

        # self.parametrise = Bdict['trainData'].shape[-1] == self.N_augment

    def printESNparameters(self):
        print('\n --------------------  ESN Parameters -------------------- ',
              '\n Data filename: {}'.format(self.filename),
              '\n Training time: {} s, \t Validation time: {} s'.format(self.t_train, self.t_val),
              '\n Washout steps: {}, \t Upsample'.format(self.N_wash, self.upsample),
              '\n Num of neurones: {}, \t Run test?: {}'.format(self.N_units, self.test_run),
              '\n Augmentat data?: {}, \t Num of training datasets: {}'.format(self.augment_data, self.L),
              '\n Connectvity: {}, \t Tikhonov parameter: {}'.format(self.connect, self.tikh),
              '\n Spectral radius: {}, \t Input scaling: {}'.format(self.rho, self.sigma_in)
              )

    def getWeights(self):  # TODO maybe
        pass

    def updateWeights(self, weights):  # TODO maybe
        pass

    def getBias(self, *args):
        return self.b

    def getReservoirState(self):
        return self.b, self.r

    def updateReservoir(self, r):
        self.hist_r = np.concatenate((self.hist_r, r))
        self.r = r[-1]

    @property
    def WCout(self):
        if not hasattr(self, '_WCout'):
            self._WCout = la.lstsq(self.Wout[:-1], self.W)[0]
        return self._WCout

    def stateDerivative(self, y):

        # Get current state
        b_in, r_in = self.getReservoirState()

        Win_1 = self.Win[:self.N_dim, :].transpose()
        Wout_1 = self.Wout[:self.N_units, :].transpose()

        # # Option(i) rin function of bin:
        # b_aug = np.concatenate((bin / self.norm, np.array([self.bias_in])))
        # rout = np.tanh(np.dot(b_aug * self.sigma_in, self.Win) + self.rho * np.dot(bin, self.WCout))
        # drout_dbin = self.sigma_in * Win_1 / self.norm + self.rho * self.WCout.transpose()

        # Option(ii) rin constant:
        rout = self.step(b_in, r_in)[1]
        drout_dbin = self.sigma_in * Win_1 / self.norm

        # Compute Jacobian
        T = 1 - rout ** 2
        J = np.dot(Wout_1, np.array(drout_dbin) * np.expand_dims(T, 1))

        return -J

    def timeIntegrate(self, t, y=None):

        # t_y = np.linspace(self.t_interp, self.t_interp + Nt * self.dt_ESN/self.upsample, Nt + 1)
        Nt = int(round(len(t) / self.upsample))
        t_b = np.linspace(self.t, self.t + Nt * self.dt_ESN, Nt + 1)

        if self.initialised:
            b, r = self.closedLoop(Nt)
        elif t[-1] < self.washout_t[-1]:
            b = np.zeros((Nt + 1, self.N_dim))
            r = np.zeros((Nt + 1, self.N_units))
        else:
            # observable washout data
            wash_obs = self.washout_obs  # truth, observables at high frequency

            # forecast model washout data
            wash_model = np.mean(y[::self.upsample], -1)
            spline = interp1d(t_b, wash_model, kind='cubic', axis=0, copy=True, fill_value=0)
            wash_model = spline(self.washout_t)

            # bias washout, the input data to open loop

            washout = wash_obs - wash_model
            # open loop initialisation of the ESN
            b_open, r_open = self.openLoop(washout)

            # do not keep the open-loop bias in the history if prefer a smooth plot
            b = np.zeros((Nt + 1, self.N_dim))
            r = np.zeros((Nt + 1, self.N_units))

            self.b, self.r = b_open[-1], r_open[-1]

            Nt_open = len(self.washout_t)

            Nt_closed = round((t_b[-1] - self.washout_t[-1]) / self.dt_ESN)
            b_closed, r_closed = self.closedLoop(Nt_closed)

            b[-(Nt_open + Nt_closed):] = np.append(b_open, b_closed[1:], axis=0)
            r[-(Nt_open + Nt_closed):] = np.append(r_open, r_closed[1:], axis=0)

            self.initialised = True

            # print('initialised bias')

            # # ESN PLOT DEBUG
            # plt.figure()
            # plt.plot(self.washout_t, washout[:, 0], '-o')
            # plt.plot(t_b[1:], b[1:, 0], '-x')
            # plt.xlim([self.washout_t[0], t_b[-1]])
            # plt.ylim([min(washout[:, 0])*1.2, max(washout[:, 0])*1.2])
            # plt.show()

        # update bias and reservoir history
        self.updateReservoir(r[1:])

        return b[1:], t_b[1:]

    def step(self, b, r):  # ________________________________________________________
        """ Advances one ESN time step.
            Returns:
                new reservoir state (no bias_out)
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)
        b_aug = np.concatenate((b / self.norm, self.bias_in))
        # Forecast the reservoir state

        r_out = np.tanh(self.Win.T.dot(b_aug * self.sigma_in) + self.W.dot(self.rho * r))
        # output bias added
        r_aug = np.concatenate((r_out, self.bias_out))
        # compute output from ESN
        b_out = np.dot(r_aug, self.Wout)
        return b_out, r_out

    def openLoop(self, b_wash):  # ____________________________________________
        """ Initialises ESN in open-loop.
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - r: time series of reservoir states
        """
        Nt = b_wash.shape[0] - 1

        r = np.empty((Nt + 1, self.N_units))
        b = np.empty((Nt + 1, self.N_dim))
        b[0], r[0] = self.getReservoirState()
        for i in range(Nt):
            b[i + 1], r[i + 1] = self.step(b_wash[i], r[i])

        return b, r

    def closedLoop(self, Nt):  # ______________________________________________
        """ Advances ESN in closed-loop.
            Input:
                - Nt: number of forecast time steps
            Returns:
                - U:  forecast time series
                - ra: time series of augmented reservoir states
        """
        Nt = int(Nt)
        r = np.empty((Nt + 1, self.N_units))
        b = np.empty((Nt + 1, self.N_dim))
        b[0], r[0] = self.getReservoirState()
        for i in range(Nt):
            b[i + 1], r[i + 1] = self.step(b[i], r[i])

        # # ESN PLOT DEBUG
        # t_b = np.linspace(self.t_interp, self.t_interp + Nt * self.dt_ESN, Nt + 1)
        # plt.plot(t_b, b[:, 0], '-+', color='green', label='closed-loop')
        # plt.legend()
        # plt.show()

        return b, r

    # TODO at some point
    # @classmethod
    # def trainESN(cls, filename, training_params):
    #     dic_params = training_params.copy()
    #     dic_params['filename'] = filename
    #     exec(open("main_training.py").read(), dic_params)
