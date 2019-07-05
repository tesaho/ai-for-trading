"""
based on udacity's ai for trading pca module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCAFactorModel(object):
    def __init__(self, num_components, annualization_factor=252):
        self.num_components = num_components   # num of pca components
        self.annualization_factor = annualization_factor    # multiple to annualize returns: daily=252
        #
        self.num_stocks_ = None
        self.model = None
        self.factor_betas_ = None
        self.factor_returns_ = None
        self.idiosyncratic_var_matrix_ = None
        self.idiosyncratic_var_vector_ = None
        self.factor_cov_matrix_ = None
        self.explained_variance_ratio_ = None
        self.common_returns_ = None
        self.residuals_ = None

    def fit_model(self, returns):

        model = PCA(n_components=self.num_components, svd_solver="full")
        model.fit(returns)

        return model

    def get_factor_betas(self, returns):
        """
        factor_betas (index=returns.columns, columns=n_components)
        """

        return pd.DataFrame(self.model.components_.T, index=returns.columns.values, \
                            columns=np.arange(1, self.model.n_components+1))

    def get_factor_returns(self, returns):
        """
        factor_returns (index=returns.index, columns=n_components)
        """

        return pd.DataFrame(self.model.transform(returns), index=returns.index, \
                            columns=np.arange(1, self.model.n_components+1))

    def get_idiosyncratic_var_matrix(self, returns, factor_returns, factor_betas):
        """
        factor_returns: df (index=return.index, columns=n_components)
        factor_betas: df (index=return.columns, columns=n_components)
        return: df (diagonal annualized variances, columns=return.columns, index=return.columns)
        """
        # idiosyncratic returns
        s = returns - self.get_common_returns(factor_returns, factor_betas)
        df = pd.DataFrame(np.diag(np.var(s)*self.annualization_factor), \
                            index=returns.columns.values, columns=returns.columns.values)

        return df

    def get_idiosyncratic_var_vector(self, idio_var_matrix):

        return pd.DataFrame(data=np.diag(idio_var_matrix.values), index=idio_var_matrix.columns.values, \
                            columns=["idio_var"])

    def get_factor_covariance_matrix(self, factor_returns):
        """
        factor_returns: df (columns=return.columns, index=return.index)
        return: df (diagonal annualized variances, columns=return.columns, index=return.columns)
        """

        df = pd.DataFrame(np.diag(np.var(factor_returns, axis=0, ddof=1) * self.annualization_factor), \
                          index=factor_returns.columns.values, columns=factor_returns.columns.values)

        return df

    def get_common_returns(self, factor_returns, factor_betas):

        return pd.DataFrame(data=np.dot(factor_returns, factor_betas.T),
                            index=factor_returns.index, columns=factor_betas.index)

    def get_residuals(self, returns, common_returns):

        return returns - common_returns

    def get_explained_variance_ratio(self, model):

        return pd.DataFrame(data=model.explained_variance_ratio_, index=np.arange(1, model.n_components+1), \
                            columns=['explained_var_ratio'])

    def fit(self, returns):

        self.num_stocks_ = len(returns.columns)
        self.model = self.fit_model(returns)
        self.factor_betas_ = self.get_factor_betas(returns)
        self.factor_returns_ = self.get_factor_returns(returns)
        self.explained_variance_ratio_ = self.get_explained_variance_ratio(self.model)
        self.common_returns_ = self.get_common_returns(self.factor_returns_, self.factor_betas_)
        self.residuals_ = self.get_residuals(returns, self.common_returns_)
        self.idiosyncratic_var_matrix_ = self.get_idiosyncratic_var_matrix(returns, self.factor_returns_, self.factor_betas_)
        self.idiosyncratic_var_vector_ = self.get_idiosyncratic_var_vector(self.idiosyncratic_var_matrix_)
        self.factor_cov_matrix_ = self.get_factor_covariance_matrix(self.factor_returns_)

    def get_factor_exposures(self, weights):
        B = self.factor_betas_.loc[weights.index]
        return B.T.dot(weights)

    def plot_explained_variance(self):

        fig = plt.figure()
        plt.bar(np.arange(self.num_components), self.model.explained_variance_ratio_)
        plt.title("Explained Variance Ratio")

        return fig

    def plot_factor_returns(self, top_n_factors=5):

        fig = plt.figure()
        self.factor_returns_.loc[:, 0:top_n_factors].cumsum().plot()
        plt.title("Top %s Factor Returns" %(top_n_factors))

        return fig