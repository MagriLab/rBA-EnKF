# Bias-EnKF

Note: this repo is not being maintained. The most recent version which includes tutorials can be found in [real-time bias aware DA](https://github.com/MagriLab/real-time-bias-aware-DA).

--------------------------------------------------

This repository was used in 

    Nóvoa, A., Racca, A. & Magri, L. (2023)
    Inferring unknown unknowns: Regularized bias-aware ensemble Kalman filter. 
    Computer Methods in Applied Mechanics and Engineering, 418, 116502.

Errata:
- The r-EnKF equations (15)-(16) in [Nóvoa et al. (2023)](https://doi.org/10.1016/j.cma.2023.116502) have a typo.
  The equations in this repository are correct. Please, see the Erratum [here](https://github.com/MagriLab/rBA-EnKF/blob/main/Erratum.pdf), which includes the corrected equations as well as a detailed derivation. 

--------------------------------------------------

Available filters:
- EnKF
- EnSRKF
- Regularized bias-aware EnKF

Available models:
- Rijke tube (dimensional with Galerkin discretisation)
- Van der Pol
 
Available bias estimator:
- Echo State Network
- NoBias
