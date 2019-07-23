"""
based on udacity's ai for trading PCA module utils
zipline == 1.3
"""

import os
import pandas as pd

from zipline.assets._assets import Equity
from zipline.data import bundles
from zipline.pipeline import Pipeline
from zipline.data.data_portal import DataPortal
from zipline.pipeline.classifiers import Classifier
from zipline.utils.calendars import get_calendar
from zipline.pipeline.data import USEquityPricing
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.pipeline.factors import AverageDollarVolume
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.numpy_utils import int64_dtype

# Specify the bundle name
bundle_name = 'm4-quiz-eod-quotemedia'
ingest_data_time_period = "daily"
start_date = "2011-01-05"
end_date = "2016-01-05"
env_folder_name = "pca_eod"
pipeline_screen = AverageDollarVolume(window_length = 120).top(500)

class PricingLoader(object):
    def __init__(self, bundle_data):
        self.loader = USEquityPricingLoader(bundle_data.equity_daily_bar_reader,
                                            bundle_data.adjustment_reader)
    def get_loader(self, column):
        if column not in USEquityPricing.columns:
            raise Exception('%s not in USEquityPricing.columns' %(column))
        return self.loader

class DataLoader(object):
    def __init__(self, pipeline_screen, bundle_name, start_date, end_date, \
                 exchange_calendar="NYSE",
                 data_frequency="daily"):
        self.pipeline_screen = pipeline_screen  # aka universe
        self.bundle_name = bundle_name
        self.exchange_calendar = exchange_calendar
        self.data_frequency = data_frequency

        # Set start_date and end_date
        self.start_date = self.get_date(start_date)
        self.end_date = self.get_date(end_date)

        # Set environment variable 'ZIPLINE_ROOT' to the path where the most recent data is located
        # ZIPLINE_ROOT = ./udacity/
        os.environ['ZIPLINE_ROOT'] = os.path.join(os.getcwd(), '..')
        print("ZIPLINE_ROOT set")

        # create ingest function
        self.ingest_func = csvdir_equities([data_frequency], bundle_name)

    def load_data(self):

        # Set the trading calendar
        self.trading_calendar = self.get_trading_calendar(self.exchange_calendar)

        # Load the data bundle
        self.bundle_data = self.get_bundle_data(self.bundle_name, self.ingest_func)
        print("load data bundle")

        # Create a Pipeline engine
        self.engine = self.get_engine(self.trading_calendar, self.bundle_data)
        print("pipeline engine created")

        # Create an empty Pipeline with the given screen
        self.pipeline = self.get_pipeline(self.pipeline_screen)
        print("pipeline created")

        # Get the values in index level 1 and save them to a list
        self.universe_tickers = self.get_tickers(self.engine, self.pipeline, self.end_date)
        print("universe_tickers: %s" %(len(self.universe_tickers)))

        # Create a data portal
        self.data_portal = self.get_data_portal(self.bundle_data, self.trading_calendar)
        print("data_portal created")

        # Get returns_df
        self.returns = self.get_returns(self.data_portal, self.trading_calendar, self.universe_tickers, \
                                        self.start_date, self.end_date, self.data_frequency)

        return self.returns

    def get_date(cls, date):
        return pd.Timestamp(date, tz="utc", offset="C")

    def get_trading_calendar(cls, exchange_calendar):
        # get trading calendar
        return get_calendar(exchange_calendar)

    def get_pipeline(cls, pipeline_screen):
        return Pipeline(screen=pipeline_screen)

    def get_bundle_data(cls, bundle_name, ingest_func):
        # register bundle
        bundles.register(bundle_name, ingest_func)
        # Load the data bundle
        return bundles.load(bundle_name)

    def get_engine(cls, trading_calendar, bundle_data):
        pricing_loader = PricingLoader(bundle_data)

        return SimplePipelineEngine(get_loader=pricing_loader.get_loader,
                            calendar=trading_calendar.all_sessions,
                            asset_finder=bundle_data.asset_finder)

    def get_tickers(cls, engine, pipeline, end_date):
        return engine.run_pipeline(pipeline, end_date, end_date).index.get_level_values(1).values.tolist()

    def get_data_portal(cls, bundle_data, trading_calendar):
        return DataPortal(bundle_data.asset_finder,
               trading_calendar=trading_calendar,
               first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
               equity_daily_reader=bundle_data.equity_daily_bar_reader,
               adjustment_reader=bundle_data.adjustment_reader)

    def get_pricing(cls, data_portal, trading_calendar, universe_tickers, start_d, end_d, data_frequency, frequency="1d", \
                    field='close'):

        # Set the given start and end dates to Timestamps. The frequency string C is used to
        # indicate that a CustomBusinessDay DateOffset is used
        end_dt = cls.get_date(end_d)
        start_dt = cls.get_date(start_d)

        # Get the locations of the start and end dates
        end_loc = trading_calendar.closes.index.get_loc(end_dt)
        start_loc = trading_calendar.closes.index.get_loc(start_dt)

        # return the historical data for the given window
        return data_portal.get_history_window(assets=universe_tickers, end_dt=end_dt, bar_count=end_loc - start_loc,
                                              frequency=frequency,
                                              field=field,
                                              data_frequency=data_frequency)

    def get_returns(cls, data_portal, trading_calendar, universe_tickers, start_date, end_date, data_frequency):
        # Get the historical data for the given window
        historical_data = cls.get_pricing(data_portal, trading_calendar, universe_tickers, start_date, end_date, data_frequency)

        return historical_data.pct_change()[1:].fillna(0)