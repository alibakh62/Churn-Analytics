import warnings
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from churn import config
from brandlogic import BrandLogic
from utils import Utils
from impute import Impute


class BrandSegment:

    @staticmethod
    def add_segment(df: DataFrame, gby_lst, brand="brand"):
        if brand == "brand":
            if config.brand_LOGIC_EXISTS:
                return BrandLogic.add_brand_segment(df, gby_lst)
            else:
                raise ValueError("brand logic for the brand doesn't exist!")
        else:
            return None

    @staticmethod
    def _balance_segments(df: DataFrame, gby_lst):
        '''
        transform data into balanced panel data
        Arguments
        ---------
            - df: unbalanced segment dataframe
        Returns
        -------
            - outdf: balanced panel dataframe
        '''
        outdf = (df.groupBy("{}".format(config.CUST_ID))
                 .pivot("{}".format(config.TIME_ID)).avg("segment"))
        outdf = Utils.melt_df(outdf, ["{}".format(config.CUST_ID)],
                              outdf.columns[1:], *gby_lst)
        return outdf

    @staticmethod
    def _filter_sparsity(df: DataFrame, min_obs=3):
        max_chunk_num = df.select(F.countDistinct(F.col("{}".format(config.TIME_ID)))
                                  .alias("total_obs"))
        max_null_allowed = max_chunk_num.collect()[0]["total_obs"] - min_obs
        tmp = df.groupBy("{}".format(config.CUST_ID)).agg(F.count(F.when(F.isnan("segment") |
                                                      F.col("segment").isNull(),
                                                      '-9999')).alias("count_null"))
        tmp = (tmp.alias("A").join(df.alias("B"),
                                   F.col("A.{}".format(config.CUST_ID)) == F.col("B.{}".format(config.CUST_ID)))
               .select("A.{}".format(config.CUST_ID), "B.{}".format(config.TIME_ID), "A.count_null"))
        if "{}".format(config.TIME_ID) in list(tmp.columns):
            join_keys = ["{}".format(config.CUST_ID), "{}".format(config.TIME_ID)]
        else:
            join_keys = ["{}".format(config.CUST_ID)]
        outdf = Utils.join_df(tmp, df, join_keys, how='inner')
        outdf = (outdf.filter(F.col("count_null") < max_null_allowed)
                 .select("{}".format(config.CUST_ID), "{}".format(config.TIME_ID), "segment"))
        return outdf

    @staticmethod
    def _impute_nulls(df: DataFrame, method='fill', impute_const=0):
        '''
        impute the null values in segment column,
        defaults to 'fill' (ffill then bfill)
        Arguments
        ---------
            - df: balanced segment df
            - method: two options now: 1)ffill 2)replace fixed value
            - impute_const: in case (2), this value is replaced with nulls
        Returns
        -------
            - outdf: same df as input with segment column imputed
        '''
        if method == 'fill':
            outdf = Impute.fill(df)
        elif method == 'ffill':
            outdf = Impute.ffill(df)
        elif method == 'bfill':
            outdf = Impute.bfill(df)
        elif method == 'fill_val':
            outdf = Impute.fill_val(df)
        else:
            outdf = df

        if outdf.select("*").where("segment is NULL").count() != 0:
            warnings.warn("There is still NULL values in data!")

        return outdf

    @classmethod
    def get_balanced_data(cls, df: DataFrame, gby_lst, impute_method, min_obs):
        outdf = cls._balance_segments(df)
        if min_obs is not None:
            outdf = cls._filter_sparsity(outdf, min_obs=min_obs)
        outdf = cls._impute_nulls(outdf, method=impute_method)
        return outdf

    @staticmethod
    def add_segment_prev(df: DataFrame, gby_lst):
        '''
        given a segmentDF, adds the 'segment_prev' (segment @ t-1) column,
        it drops the t=0 rows since 'segment_prev' is null (not known)
        Arguments
        ---------
            - df: imputed segment df,
                  must have these columns: "memberid", "chunkid", "segment"
        Returns
        -------
            - outdf: adds previous segment column to the input df
        '''
        w = (Window.partitionBy("{}".format(config.CUST_ID))
             .orderBy("{}".format(config.TIME_ID)))
        outdf = (df.withColumn("segment_prev", F.lag("segment", 1).over(w))
                 .orderBy(*gby_lst)
                 )
        outdf = outdf.na.drop('any')

        return outdf
