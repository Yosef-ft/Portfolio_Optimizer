import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cvxpy

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import CLA, objective_functions



class PortOpt:

    def portfolio_performance(
        self,weights, expected_returns, cov_matrix, verbose=True, risk_free_rate=0.02
    ):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

        :param expected_returns: expected returns for each asset. Can be None if
                                optimising for volatility only (but not recommended).
        :type expected_returns: np.ndarray or pd.Series
        :param cov_matrix: covariance of returns for each asset
        :type cov_matrix: np.array or pd.DataFrame
        :param weights: weights or assets
        :type weights: list, np.array or dict, optional
        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calculated yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        if isinstance(weights, dict):
            if isinstance(expected_returns, pd.Series):
                tickers = list(expected_returns.index)
            elif isinstance(cov_matrix, pd.DataFrame):
                tickers = list(cov_matrix.columns)
            else:
                tickers = list(range(len(expected_returns)))
            new_weights = np.zeros(len(tickers))

            for i, k in enumerate(tickers):
                if k in weights:
                    new_weights[i] = weights[k]
            if new_weights.sum() == 0:
                raise ValueError("Weights add to zero, or ticker names don't match")
        elif weights is not None:
            new_weights = np.asarray(weights)
        else:
            raise ValueError("Weights is None")

        sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))

        if expected_returns is not None:
            mu = objective_functions.portfolio_return(
                new_weights, expected_returns, negative=False
            )

            sharpe = objective_functions.sharpe_ratio(
                new_weights,
                expected_returns,
                cov_matrix,
                risk_free_rate=risk_free_rate,
                negative=False,
            )
            if verbose:
                st.subheader('Portfolio Performance')
                st.write("Expected annual return: {:.1f}%".format(100 * mu))
                st.write("Annual volatility: {:.1f}%".format(100 * sigma))
                st.write("Sharpe Ratio: {:.2f}".format(sharpe))
            return mu, sigma, sharpe
        else:
            if verbose:
                print("Annual volatility: {:.1f}%".format(100 * sigma))
            return None, sigma, None


    def _plot_ef(self, ef, ef_param, ef_param_range, ax, show_assets, show_tickers):
        """
        Helper function to plot the efficient frontier from an EfficientFrontier object
        """
        mus, sigmas = [], []

        # Create a portfolio for each value of ef_param_range
        for param_value in ef_param_range:
            if ef_param == "utility":
                ef.max_quadratic_utility(param_value)
            elif ef_param == "risk":
                ef.efficient_risk(param_value)
            elif ef_param == "return":
                ef.efficient_return(param_value)
            else:
                raise NotImplementedError(
                    "ef_param should be one of {'utility', 'risk', 'return'}"
                )

            ret, sigma, _ = ef.portfolio_performance()
            mus.append(ret)
            sigmas.append(sigma)

        ax.plot(sigmas, mus, label="Efficient frontier")

        asset_mu = ef.expected_returns
        asset_sigma = np.sqrt(np.diag(ef.cov_matrix))
        if show_assets:
            ax.scatter(
                asset_sigma,
                asset_mu,
                s=30,
                color="k",
                label="assets",
            )
            if show_tickers:
                for i, label in enumerate(ef.tickers):
                    ax.annotate(label, (asset_sigma[i], asset_mu[i]))
        return ax    


    def _ef_default_returns_range(self,ef, points):
        """
        Helper function to generate a range of returns from the GMV returns to
        the maximum (constrained) returns
        """
        ef_minvol = ef.deepcopy()
        ef_maxret = ef.deepcopy()

        ef_minvol.min_volatility()
        min_ret = ef_minvol.portfolio_performance()[0]
        max_ret = ef_maxret._max_return()
        return np.linspace(min_ret, max_ret - 0.0001, points)    

    def _plot_cla(self,cla, points, ax, show_assets, show_tickers):
        """
        Helper function to plot the efficient frontier from a CLA object
        """
        if cla.weights is None:
            cla.max_sharpe()
        optimal_ret, optimal_risk, _ = cla.portfolio_performance()

        if cla.frontier_values is None:
            cla.efficient_frontier(points=points)

        mus, sigmas, _ = cla.frontier_values

        ax.plot(sigmas, mus, label="Efficient frontier")
        ax.scatter(optimal_risk, optimal_ret, marker="x", s=100, color="r", label="optimal")

        asset_mu = cla.expected_returns
        asset_sigma = np.sqrt(np.diag(cla.cov_matrix))
        if show_assets:
            ax.scatter(
                asset_sigma,
                asset_mu,
                s=30,
                color="k",
                label="assets",
            )
            if show_tickers:
                for i, label in enumerate(cla.tickers):
                    ax.annotate(label, (asset_sigma[i], asset_mu[i]))
        return ax    

    def plot_efficient_frontier(
        self, 
        opt,
        ef_param="return",
        ef_param_range=None,
        points=100,
        ax=None,
        show_assets=True,
        show_tickers=True,
        **kwargs
    ):
        """
        Plot the efficient frontier based on either a CLA or EfficientFrontier object.

        :param opt: an instantiated optimizer object BEFORE optimising an objective
        :type opt: EfficientFrontier or CLA
        :param ef_param: [EfficientFrontier] whether to use a range over utility, risk, or return.
                        Defaults to "return".
        :type ef_param: str, one of {"utility", "risk", "return"}.
        :param ef_param_range: the range of parameter values for ef_param.
                            If None, automatically compute a range from min->max return.
        :type ef_param_range: np.array or list (recommended to use np.arange or np.linspace)
        :param points: number of points to plot, defaults to 100. This is overridden if
                    an `ef_param_range` is provided explicitly.
        :type points: int, optional
        :param show_assets: whether we should plot the asset risks/returns also, defaults to True
        :type show_assets: bool, optional
        :param show_tickers: whether we should annotate each asset with its ticker, defaults to False
        :type show_tickers: bool, optional
        :param filename: name of the file to save to, defaults to None (doesn't save)
        :type filename: str, optional
        :param showfig: whether to plt.show() the figure, defaults to False
        :type showfig: bool, optional
        :return: matplotlib axis
        :rtype: matplotlib.axes object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = ax or plt.gca()

        if isinstance(opt, CLA):
            ax = self._plot_cla(
                opt, points, ax=ax, show_assets=show_assets, show_tickers=show_tickers
            )
        elif isinstance(opt, EfficientFrontier):
            if ef_param_range is None:
                ef_param_range = self._ef_default_returns_range(opt, points)

            ax = self._plot_ef(
                opt,
                ef_param,
                ef_param_range,
                ax=ax,
                show_assets=show_assets,
                show_tickers=show_tickers,
            )
        else:
            raise NotImplementedError("Please pass EfficientFrontier or CLA object")

        ax.legend()
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        st.subheader("Efficient Frontier")
        st.pyplot(fig)
        # self._plot_io(**kwargs)
        return ax


    def calculate_eReturn_covariance(self, adj_close: pd.DataFrame):
        '''
        This function calculates the covariance matrix and expected return for a given adjusted price.

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
        
        Returns:
            covariance_matrix(pandas.DatFrame), Expected_return(pandas.Series)
        '''

        expected_return = mean_historical_return(adj_close)
        covariance_matrix = CovarianceShrinkage(adj_close).ledoit_wolf()

        return covariance_matrix, expected_return
    

    def calculate_EfficientFrontier(self, adj_close: pd.DataFrame):
        '''
        This fuction calculates the efficient frontier and also the weights

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers

        Returns:
            efficient_frontier (pypfopt.efficient_frontier.efficient_frontier.EfficientFrontier)
        '''

        covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
        ef = EfficientFrontier(expected_return, covariance_matrix)

        return ef

        
    def clean_weights(self, adj_close: pd.DataFrame):
        '''
        This function calculates the weights and returns the clean weight for your portfolio

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
        
        Returns:
            clean_weight(OrderedDict)
        '''

        covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
        ef = EfficientFrontier(expected_return, covariance_matrix)

        weights = ef.max_sharpe()
        clean_weight = ef.clean_weights()

        return clean_weight
    

    def plot_weights(self, adj_close: pd.DataFrame, allow_shorts=False):
        '''
        This function takes Efficient Frontier and plots the weights

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            allow_shorts(bool): this will plot if we allow short selling

        Returns:
            matplotlib plot
        '''

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1)) 
            
        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
             

        plotting.plot_weights(ef.max_sharpe())


    def plot_efficient_frontier_custom(self, adj_close: pd.DataFrame, allow_shorts=False):
        '''
        This function takes Efficient Frontier and plots the Efficient frontier

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            allow_shorts(bool): this will plot if we allow short selling

        Returns:
            matplotlib plot
        ''' 

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
            ef_plot = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1))

            weights_plot = ef_plot.max_sharpe()

            ef_plot.portfolio_performance(verbose=True)

            ef_constraints = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1))

            ef_constraints.add_constraint(lambda x: cvxpy.sum(x) == 1)

            fig, ax = plt.subplots()
            ax.scatter(ef_plot.portfolio_performance()[1], ef_plot.portfolio_performance()[0], marker='*', color='r', s=200,
                    label='Tangency portfolio')

            ax.legend()
            self.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            # plt.show()
            # st.pyplot(ax)

        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
            ef_plot = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))

            weights_plot = ef_plot.max_sharpe()

            ef_plot.portfolio_performance(verbose=True)

            ef_constraints = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))

            ef_constraints.add_constraint(lambda x: cvxpy.sum(x) == 1)

            fig, ax = plt.subplots()
            ax.scatter(ef_plot.portfolio_performance()[1], ef_plot.portfolio_performance()[0], marker='*', color='r', s=200,
                    label='Tangency portfolio')

            ax.legend()
            self.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            # plt.show()
            # st.pyplot(ax)


    def budget_allocator(self, stocks,ava_money ,allow_shorts=False):



        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(stocks)
            ef = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1)) 
            
        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(stocks)
            ef = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))   

        w = ef.max_sharpe()
        latest_prices = get_latest_prices(stocks)
        da = DiscreteAllocation(w, latest_prices, total_portfolio_value=ava_money)
        allocation, leftover = da.lp_portfolio()

        self.portfolio_performance(w, expected_return, covariance_matrix)
        return allocation, leftover
    

