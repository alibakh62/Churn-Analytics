from collections import Counter
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from churn import config
from utils import Utils
from brandlogic import BrandLogic
from rfm import RFM


class FeatureExtraction:

    groupby_cols = config.GROUPBY_COLS
    most_common_cols = config.MOSTCOMMON_COLS

    @property
    def params(cls):
        return cls.groupby_cols, cls.most_common_cols

    @params.setter
    def params(cls, groupby_cols, most_common_cols):
        cls.groupby_cols = groupby_cols
        cls.most_common_cols = most_common_cols

    @staticmethod
    def add_basket_desc(df: DataFrame, gby_lst, trans_amt_col, trans_vol_col):
        outdf = df.groupBy(*gby_lst).agg(F.avg(trans_amt_col)
                                         .alias("avg_basket_size"),
                                         F.sum(trans_amt_col)
                                         .alias("tot_basket_size"),
                                         F.countDistinct(trans_vol_col)
                                         .alias("tot_num_visits"))
        return outdf

    @staticmethod
    def add_hour_of_day(df: DataFrame, gby_lst, hourofday_col):
        outdf = df.groupBy(*gby_lst).agg(F.avg(hourofday_col)
                                         .alias("avg_visit_time"))
        return outdf

    @staticmethod
    def add_dow(df: DataFrame, gby_lst, dow_col, day):
        outdf = (df.filter("{0}='{1}'".format(dow_col, day)).groupBy(*gby_lst)
                 .agg(F.countDistinct(config.TRANS_ORD_COL)
                      .alias("num_visits_{}".format(day))
                      )
                 )
        return outdf

    @staticmethod
    def add_most_common(df: DataFrame, gby_lst, mostcommon_cols):
        @F.udf
        def mode(x):
            return Counter(x).most_common(1)[0][0]

        agg_expr = [mode(F.collect_list(col)).alias(col+"_mostCommon")
                    for col in mostcommon_cols]
        outdf = df.groupBy(*gby_lst).agg(*agg_expr)

        return outdf

    @staticmethod
    def add_rfm_feature(df: DataFrame, gby_lst):
        return RFM().get_rfm_segment(df, gby_lst)

    @classmethod
    def add_features(cls, df: DataFrame, gby_lst=config.GROUPBY_COLS):
        outdf = df

        if config.BASKET_DESC:
            tmpdf = cls.add_basket_desc(df=outdf,
                                        gby_lst=gby_lst,
                                        trans_amt_col=config.TRANS_AMT_COL,
                                        trans_vol_col=config.TRANS_ORD_COL)
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.HOUR_OF_DAY:
            tmpdf = cls.add_hour_of_day(df=outdf,
                                        gby_lst=gby_lst,
                                        hourofday_col=config.HOUR_OF_DAY_COL)
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.DAY_OF_WEEK:
            days_lst = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                        'Saturday', 'Sunday']
            for day in days_lst:
                tmpdf = cls.add_dow(df=outdf,
                                    gby_lst=gby_lst,
                                    dow_col=config.DAY_OF_WEEK_COL,
                                    day=day)
                outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.MOST_COMMON:
            tmpdf = cls.add_most_common(df=outdf,
                                        gby_lst=gby_lst,
                                        mostcommon_cols=config.MOSTCOMMON_COLS)
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.RFM_FEATURE:
            tmpdf = cls.add_rfm_feature(df=outdf,
                                        gby_lst=gby_lst)
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)
            outdf = outdf.fillna({"RFMScore": 555})

        if config.GET_BRND_FEATURES:
            tmpdf = BrandLogic.add_brand_features(df, gby_lst)
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        return outdf

        @staticmethod
        def add_delta_features(df: DataFrame, gby_lst, delta_cols_lst):
            '''
            calculates t-(t-1) values for feature values
            Arguments
            ---------
                - df: input df, including the features
            Returns
            -------
                - outdf: same as input df with addition of delta features
            '''
            w = (Window.paritionBy("{}".format(config.CUST_ID))
                 .orderBy("{}".format(config.TIME_ID)))
            outdf = df
            for dltcol in delta_cols_lst:
                outdf = (outdf.withColumn("{}_prev".format(dltcol),
                                          F.lag(dltcol, 1).over(w))
                         .orderBy(*gby_lst)
                         )
                outdf = outdf.withColumn("{}_delta".format(dltcol),
                                         (F.col(dltcol) - F.col(dltcol+"_prev"))
                                         / F.col(dltcol+"_prev"))
            outdf = outdf.na.drop('any')
            return outdf
