"""
Digital Twin drift-prior model (reframed from the original real-time
hardware-control architecture into a search-space prior for image-based
recovery).

Predicts an expected drift RADIUS (not vector -- direction is not
recoverable from elapsed time alone in this design, see note below) of the
true site from the nominal recorded coordinate, using only elapsed_time_s:
legitimate metadata any inspection tool logs, no physical telemetry needed.

Functional form mirrors the original doc's thermal reduced-order model
(dL = alpha * L * dT, saturating exponential), but with ESTIMATED
population-level parameters that start mis-calibrated and are refined by
Loop 1 as ground-truth match error accumulates across sites.
"""
import numpy as np


class DriftPrior:
    def __init__(self, d_max_est=150.0, tau_est=3000.0, base_uncertainty_px=90.0):
        self.d_max_est = d_max_est
        self.tau_est = tau_est
        self.base_uncertainty_px = base_uncertainty_px
        self.n_updates = 0
        self.history = []  # (elapsed_s, error) for diagnostics
        self.history_obs = []  # (elapsed_s, observed_drift_mag) for tau refitting

    def predict(self, elapsed_time_s):
        """Returns (radius_px, uncertainty_px). The true site is expected to
        lie within roughly radius +/- uncertainty of the nominal coordinate,
        direction unknown -- an annulus prior, not a point prior."""
        radius = self.d_max_est * (1 - np.exp(-elapsed_time_s / self.tau_est))
        uncertainty = self.base_uncertainty_px / (1 + 0.15 * self.n_updates) ** 0.5
        return float(radius), float(uncertainty)

    def _refit_params(self):
        if len(self.history_obs) < 8:
            return
        ts = np.array([t for t, d in self.history_obs])
        ds = np.array([d for t, d in self.history_obs])
        try:
            from scipy.optimize import curve_fit
            def model(t, d_max, tau):
                return d_max * (1 - np.exp(-t / tau))
            popt, _ = curve_fit(model, ts, ds, p0=[self.d_max_est, self.tau_est],
                                bounds=([20.0, 500.0], [400.0, 10000.0]))
            self.d_max_est, self.tau_est = float(popt[0]), float(popt[1])
        except Exception:
            pass

    def update(self, elapsed_time_s, observed_drift_mag):
        """Loop 1 (EKF-lite): nudge d_max_est toward explaining the observed
        drift magnitude at this elapsed time. Online, one sample at a time --
        mirrors the Kalman-filter feedback role in the original architecture,
        just applied to a scalar radius instead of a full physics state."""
        predicted_radius, _ = self.predict(elapsed_time_s)
        error = observed_drift_mag - predicted_radius
        lr = 0.15
        saturation = (1 - np.exp(-elapsed_time_s / self.tau_est))
        if saturation > 1e-3:
            self.d_max_est += lr * error * saturation
        self.d_max_est = float(np.clip(self.d_max_est, 20.0, 400.0))
        self.n_updates += 1
        self.history.append((elapsed_time_s, error))
        self.history_obs.append((elapsed_time_s, observed_drift_mag))
        if self.n_updates % 5 == 0:
            self._refit_params()
        return error
