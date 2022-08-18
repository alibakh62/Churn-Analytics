import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import LongType, StringType, IntegerType
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
from churn import config
from brandlogic import BrandLogic
from feature_extraction import FeatureExtraction
from brandsegment import BrandSegment
from utils import Utils


class DataLoader:

    window = 3
    step = 1
    impute = 'fill'

    @property
    def params(cls):
        return cls.window, cls.step, cls.impute

    @params.setter
    def params(cls, window, step, impute):
        cls.window = window
        cls.step = step
        cls.impute = impute

    @staticmethod
    def load_raw_data(brand="brand"):
        if brand == "brand":
            if config.brand_LOGIC_EXISTS:
                return BrandLogic.load_brand_input_data()
            else:
                raise ValueError("brand logic for the brand doesn't exist!")
        else:
            return None

    @staticmethod
    def get_data_chunk(df: DataFrame, start_date=None, window=3):
        '''
        Filters input data given start date and window size
        Arguments
        ---------
            - df: customer transaction data
            - start_date: start date of window
            - window: number of months in window
        Returns
        -------
            - outdf: customer transaction data filtered for window
        '''
        if start_date is None:
            start_date = config.START_DATE
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') +
                    relativedelta(months=+window)
                    )
        end_date = datetime.strftime(end_date, '%Y-%m-%d')

        outdf = (df
                 .withColumnRenamed("{}".format(config.TRANS_DATE), "transdate")
                 .withColumn("transdate", F.date_format('transDate',
                                                        "yyyy-MM-dd"))
                 .filter((F.col("transdate") >= start_date) &
                         (F.col("transdate") < end_date))
                 )
        # adding chunk unique ID
        outdf = (outdf
                 .withColumn("{}".format(config.TIME_ID), F.lit(start_date))
                 .withColumn("{}".format(config.TIME_ID),
                             F.regexp_replace(F.col("{}".format(config.TIME_ID))
                                              .cast(StringType()), "-", ""))
                 )
        return outdf

    @classmethod
    def gen_chunks(cls, df: DataFrame, start_date=None, window=3, step=1):
        '''
        Generates data chunks with a rolling window
        Arguments
        ---------
            - df: customer transaction data
            - start_date: start date of window
            - window: number of months in window
            - step: number of months to roll by
        Returns
        -------
            - customer transaction data filtered for window
        '''
        if start_date is None:
            start_date = config.START_DATE

        while start_date < config.END_DATE:
            outdf = cls.get_data_chunk(df, start_date)
            start_date = datetime.strptime(start_date, '%Y-%m-%d') \
                + relativedelta(months=+step)
            start_date = datetime.strftime(start_date, '%Y-%m-%d')
            yield outdf

    @classmethod
    def get_segment(cls, df: DataFrame, gby_lst, rolling):
        '''
        Stacks up all data chunks after adding segment column to each
        Arguments
        ---------
            - df: customer transaction data with history columns
        Returns
        -------
            - outdf: dataframe contains memberid, (chunkid), segment
        '''
        spark = SparkSession.builder.getOrCreate()
        outdf = BrandSegment.add_segment(df.withColumn("chunkid", F.lit("999")),
                                         gby_lst=config.GROUPBY_COLS).cache()
        outdf.count()
        if rolling:
            schema = StructType([StructField("{}".format(config.CUST_ID),
                                             LongType(), True),
                                 StructField("{}".format(config.TIME_ID),
                                             StringType(), True),
                                 StructField("segment", IntegerType(), True)])
            outdf = spark.createDataFrame([], schema)
            chunkgen = cls.gen_chunks(df, window=cls.window, step=cls.step)
            while True:
                try:
                    chunkdf = BrandSegment.add_segment(next(chunkgen),
                                                       gby_lst=config.GROUPBY_COLS).cache()
                    chunkdf.count()
                    outdf = outdf.union(chunkdf)
                except StopIteration:
                    break
            outdf.count()
            outdf = BrandSegment.get_balanced_data(outdf,
                                                   gby_lst=config.GROUPBY_COLS,
                                                   impute_method=cls.impute,
                                                   min_obs=3).cache()
            outdf.count()
            outdf = BrandSegment.add_segment_prev(outdf,
                                                  gby_lst=config.GROUPBY_COLS)
        return outdf

    @classmethod
    def get_features(cls, df: DataFrame, gby_lst, rolling):
        spark = SparkSession.builder.getOrCreate()
        outdf = FeatureExtraction.add_features(df.withColumn("chunkid", F.lit("999")),
                                               gby_lst=config.GROUPBY_COLS)
        if rolling:
            schema = outdf.schema
            outdf = spark.createDataFrame([], schema)
            chunkgen = cls.gen_chunks(df, window=cls.window, step=cls.step)
            while True:
                try:
                    chunkdf = FeatureExtraction.add_features(next(chunkgen),
                                                             gby_lst=config.GROUPBY_COLS).cache()
                    chunkdf.count()
                    outdf = outdf.union(chunkdf)
                except StopIteration:
                    break

        return outdf

    @classmethod
    def transform(cls, gby_lst=config.GROUPBY_COLS):
        df = cls.load_raw_data().cache()
        df.count()
        if config.GET_BRND_SEGMENT:
            outdf = cls.get_segment(df, gby_lst).cache()
            outdf.count()
        elif config.GET_FEATURES:
            outdf = cls.get_features(df, gby_lst).cache()
            outdf.count()
        elif (config.GET_BRND_SEGMENT) & (config.GET_FEATURES):
            outdf1 = cls.get_segment(df, gby_lst).cache()
            outdf1.count()
            outdf2 = cls.get_features(df, gby_lst).cache()
            outdf2.count()
            if "{}".format(config.TIME_ID) in outdf1.columns:
                join_keys = ["{}".format(config.CUST_ID), "{}".format(config.TIME_ID)]
            else:
                join_keys = ["{}".format(config.CUST_ID)]
            outdf = Utils.join_df(outdf1, outdf2, join_keys)
        else:
            outdf = df
            warnings.warn("No transformation requested, Returning original data!")
        return outdf
