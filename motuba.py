#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file    :   motuba.py
@author  :   Chris Fonnesbeck
@desc    :   General purpose Marcel-like projection model using PyMC
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class Motuba:
    """
    General purpose Marcel-like projection model using PyMC

    For player $i$ having observed target values in three consecutive years :math:`y_i^{(t)}, y_i^{(t-1)}, y_i^{(t-2)}` we are predicting :math:`y_i^{(t+1)}` via:

    .. math::
        \theta_i^{(t+1)} = \mu_i + w_0 (\theta_i^{(t)}) + w_1 (\theta_i^{(t-1)}) + w_2 (\theta_i^{(t-2)}) + \beta_{28} a_i^{(t+1)}

    where :math:`\{w_0, w_1, w_2\}\sim \text{Dirichlet}(\phi)` are constrained to be ordered and the :math:`\theta_i` are partially pooled observations to regress extreme values:

    .. math::
        y_i^{(t)} \sim N(\theta_i^{(t)}, \sigma)

    The coefficient :math:`\beta_{28}` is corresponds to a simple triangular aging model that is centered on the passed value `peak_age`.

    All inputs are standardized to aid in convergence.

    Parameters
    ----------
    model_type : str
        Model type to use. Valid options are "normal", "binomial", and "poisson" (default: "normal")
    peak_age : int
        Age at which to center triangular aging model; if set to `None` then it is estiamted from data (defaults to 28)

    """

    def __init__(self, model_type: str = "normal", peak_age: int = 28):
        self.peak_age_ = peak_age
        self.trace_ = None

        if model_type.lower() == "normal":
            self._normal_model()
            self._scaled = True
            self._dtype = float
        elif model_type.lower() == "binomial":
            self._binomial_model()
            self._scaled = False
            self._dtype = int
        elif model_type.lower() == "poisson":
            self._poisson_model()
            self._scaled = False
            self._dtype = int
        else:
            raise ValueError("Invalid model type:", model_type)

    def _binomial_model(self):
        with pm.Model() as self.model:
            # Initialize with dummy values
            X = pm.MutableData("X", np.empty((1, 3), dtype=int))
            y = pm.MutableData("y", np.empty((1,), dtype=int))
            N_x = pm.MutableData("N_x", np.ones((1, 3), dtype=int))
            N_y = pm.MutableData("N_y", np.ones((1,), dtype=int))
            age = pm.MutableData("age", np.ones((1,), dtype=int))

            # Empirical rates
            mu = pm.Normal("mu", 0, 5)
            tau = pm.HalfNormal("tau", 3)
            z = pm.Normal("z", 0, 1, shape=X.shape)
            p = pm.math.invlogit(mu + tau * z)
            rate_like = pm.Binomial("rate_like", n=N_x, p=p, observed=X)

            # Marcel weights
            w = pm.Dirichlet(
                "w",
                a=np.array([0.8, 1.0, 1.2]),
                initval=np.array([0.2, 0.3, 0.5]),
            )

            # Coefficient for quadratic aging
            beta = pm.Normal("beta", 0.0, sigma=1.0)

            if self.peak_age_ is None:
                peak_age = pm.Uniform("peak_age", age.min(), age.max())
            else:
                peak_age = self.peak_age_

            # Sum of weighted & regressed years and aging factor
            prediction = pm.Deterministic(
                "prediction",
                pm.math.invlogit(pm.math.logit(pm.math.dot(X / N_x, w)) + beta * (age - peak_age)),
            )

            pm.Potential("shrinkage", pm.logp(pm.Normal.dist(mu, tau), pm.math.logit(prediction)))

            pm.Binomial("prediction_like", n=N_y, p=prediction, observed=y)

    def _poisson_model(self):
        with pm.Model() as self.model:
            # Initialize with dummy values
            X = pm.MutableData("X", np.empty((1, 3), dtype=int))
            y = pm.MutableData("y", np.empty((1,), dtype=int))
            N_x = pm.MutableData("N_x", np.ones((1, 3), dtype=int))
            N_y = pm.MutableData("N_y", np.ones((1,), dtype=int))
            age = pm.MutableData("age", np.ones((1,), dtype=int))

            # Partial pooling of observed values
            mu = pm.Normal("mu", 0, 5)
            tau = pm.HalfNormal("tau", 3)
            z = pm.Normal("z", 0, 1, shape=X.shape)
            theta = mu + z * tau
            pm.Poisson(
                "theta_like",
                mu=N_x * pm.math.exp(theta),
                observed=X,
            )

            # Marcel weights
            w = pm.Dirichlet(
                "w",
                a=np.array([0.8, 1.0, 1.2]),
                initval=np.array([0.2, 0.3, 0.5]),
            )

            # Coefficient for triangular aging
            beta = pm.Normal("beta", 0.0, sigma=1.0)

            if self.peak_age_ is None:
                peak_age = pm.Uniform("peak_age", age.min(), age.max())
            else:
                peak_age = self.peak_age_

            # Sum of weighted & regressed years and aging factor
            prediction = pm.Deterministic(
                "prediction",
                pm.math.exp(pm.math.dot(X, w) + beta * (age - peak_age)),
            )

            pm.Potential("shrinkage", pm.logp(pm.Normal.dist(mu, tau), pm.math.log(prediction)))

            pm.Poisson("prediction_like", mu=N_y * prediction, observed=y)

    def _normal_model(self):
        with pm.Model() as self.model:
            # Initialize with dummy values
            X = pm.MutableData("X", np.empty((1, 3)))
            y = pm.MutableData("y", np.empty((1,)))
            N_x = pm.MutableData("N_x", np.ones((1, 3), dtype=int))
            N_y = pm.MutableData("N_y", np.ones((1,), dtype=int))
            age = pm.MutableData("age", np.ones((1,), dtype=int))

            # Partial pooling of observed values
            mu = pm.Normal("mu", 0, sigma=10)
            tau = pm.Uniform("tau", 0, 10)
            z = pm.Normal("z", 0, 1, shape=X.shape)
            theta = mu + z * tau
            sigma = pm.HalfNormal("sigma", 10)
            pm.Normal(
                "theta_like",
                mu=theta,
                sigma=sigma / pm.math.sqrt(N_x),
                observed=X,
            )

            # Marcel weights
            w = pm.Dirichlet(
                "w",
                a=np.array([8, 10, 12]),
                initval=np.array([0.2, 0.3, 0.5]),
            )

            # Coefficient for triangular aging
            beta = pm.Normal("beta", 0.0, sigma=1.0)

            if self.peak_age_ is None:
                peak_age = pm.Uniform("peak_age", age.min(), age.max())
            else:
                peak_age = self.peak_age_

            # Sum of weighted & regressed years and aging factor
            prediction = pm.Deterministic(
                "prediction",
                pm.math.dot(X, w) + beta * (age - peak_age),
            )

            pm.Potential("shrinkage", pm.logp(pm.Normal.dist(mu, tau), prediction))

            pm.Normal(
                "prediction_like",
                mu=prediction,
                sigma=sigma / pm.math.sqrt(N_y),
                observed=y,
            )

    def fit(
        self,
        X: pd.DataFrame,
        N_x: pd.DataFrame,
        y: pd.Series,
        N_y: pd.Series,
        age_col: str = "age",
        n_samples=1000,
        tune=3000,
        target_accept=0.8,
        random_seed=42,
        **pymc_kwargs,
    ):
        """
        Fit model using MCMC

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe should include 3 columns of past performance and a column of age corresponding to the current year.
            The index of the DataFrame should be the `bam_id`s of the corresponding players.
        N_x : pd.DataFrame
            3-column dataframe of sample sizes corresponding to the observations in `X`.
        y : pd.Series
            Series of target values corresponding to next-year performance.
        N_y : pd.DataFrame
            Series of sample sizes corresponding to the observations in `y`.
        age_col : str
            Name of column in `X` containing age values (defaults to "age").
        n_samples : int
            Number of samples to draw from PyMC model (defaults to 1000).
        tune : int
            Number of tuning samples to draw from PyMC model (defaults to 2000).
        target_accept : float
            Target acceptance rate for NUTS algorithm (defaults to 0.9).
        pymc_kwargs : dict
            Optional keyword arguments to pass to PyMC model.
        """
        assert age_col in X.columns, f"{age_col} must be in DataFrame X"

        X = X.copy().astype(self._dtype)
        y = y.copy().astype(self._dtype)
        age = X.pop(age_col).values.astype(int)

        assert X.shape == N_x.shape
        assert y.shape == N_y.shape
        assert not X.isna().sum().any()
        assert not N_x.isna().sum().any()

        if not (X.index == y.index).all():
            print("Indexes in X and y are not the same. Make sure input and output rows correspond.")
        if not (X.index == N_x.index).all():
            print("Indexes in X and N_x are not the same. Make sure input and output rows correspond.")
        if not (y.index == N_y.index).all():
            print("Indexes in y and N_y are not the same. Make sure input and output rows correspond.")

        if self._scaled:
            self.loc_ = X.values.mean()
            self.scale_ = X.values.std()

            X = (X.values - self.loc_) / self.scale_
            y = (y.values - self.loc_) / self.scale_
        else:
            X = X.values
            y = y.values

        self.fit_data_ = {
            "X": X,
            "y": y,
            "age": age,
            "N_x": N_x.values,
            "N_y": N_y.values,
        }

        with self.model:
            pm.set_data(self.fit_data_)
            self.trace_ = pm.sample(n_samples, tune=tune, target_accept=target_accept, random_seed=random_seed, **pymc_kwargs)

        return self

    def predict(
        self,
        X: pd.DataFrame,
        N_x: pd.DataFrame,
        N_y: pd.Series,
        age_col: str = "age",
        pct_interval: float = 0.9,
        round_to: int = "none",
    ):
        """
        Predict using fitted model

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe should include 3 columns of past performance and a column of age corresponding to the current year.
            The index of the DataFrame should be the `bam_id`s of the corresponding players.
        N_x : pd.DataFrame
            3-column dataframe of sample sizes corresponding to the observations in `X`.
        N_y : pd.Series or int
            Series of sample sizes or single sample size corresponding to prediction.
        age_col : str
            Name of column in `X` containing age values (defaults to "age").
        pct_interval: float
            Percentile interval to return (defaults to 0.9, i.e. returns 5th and 95th percentile).
        round_to: int
            Number of decimal places to round predictions to (defaults to "none").

        Returns
        -------
        arviz.InferenceData
        """
        X = X.copy().astype(self._dtype)
        age = X.pop(age_col).values.astype(int)

        assert X.shape == N_x.shape
        assert not X.isna().sum().any()
        assert not N_x.isna().sum().any()

        if self._scaled:
            X_values = (X.values - self.loc_) / self.scale_
        else:
            X_values = X.values

        with self.model:
            pm.set_data(
                {
                    "X": X_values,
                    "age": age,
                    "N_x": N_x.values,
                    "N_y": N_y.values,
                    "y": np.zeros_like(N_y),
                }
            )

            ppc_samples = pm.sample_posterior_predictive(self.trace_, var_names=["prediction"])

        self.trace_.extend(ppc_samples)

        predicted_values = ppc_samples.posterior_predictive["prediction"]

        if self._scaled:
            predicted_values = (predicted_values * self.scale_) + self.loc_

        summary_table = az.summary(predicted_values, hdi_prob=pct_interval, round_to=round_to).iloc[:, :4]
        summary_table.index = X.index
        return summary_table

    def get_weights(self):
        """
        Return estimates of input weights
        """

        return az.summary(self.trace_, var_names=["w"])[["mean", "sd"]]

    def check_convergence(self):
        """
        Graphical check of model convergence
        """
        return az.plot_energy(self.trace_)

    def check_calibration(self):
        """
        Graphical check of calibration
        """
        with self.model:
            pm.set_data(self.fit_data_)
            ppc_samples = pm.sample_posterior_predictive(self.trace_)
        return az.plot_ppc(ppc_samples, var_names=["prediction_like"])
