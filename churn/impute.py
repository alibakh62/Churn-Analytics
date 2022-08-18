import sys
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from churn import config


class Impute:

    @staticmethod
    def ffill(df):
        w = (Window
             .partitionBy("{}".format(config.CUST_ID))
             .orderBy("{}".format(config.TIME_ID))
             .rowsBetween(-sys.maxsize, 0)
             )
        df = (df
              .withColumn("segment", F.last('segment', True).over(w))
              .orderBy("{}".format(config.CUST_ID),
                       "{}".format(config.TIME_ID))
              )
        return df

    @staticmethod
    def bfill(df):
        w = (Window
             .partitionBy("{}".format(config.CUST_ID))
             .orderBy("{}".format(config.TIME_ID))
             .rowsBetween(0, sys.maxsize)
             )
        df = (df
              .withColumn("segment", F.first('segment', True).over(w))
              .orderBy("{}".format(config.CUST_ID),
                       "{}".format(config.TIME_ID))
              )
        return df

    @classmethod
    def fill(cls, df):
        df = cls.ffill(df)
        df = cls.bfill(df)
        return df

    @staticmethod
    def fill_val(df, val):
        df = df.fillna(val)
        return df
