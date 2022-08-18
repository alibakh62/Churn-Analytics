import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from brandlogic import BrandLogic
from utils import Utils
from churn import config


class RFM:

    def __init__(self, by_brand=True):
        self.by_brand = by_brand

    def _count_columns(self, df):
        return df.agg(*[F.count(c).alias(c) for c in df.columns]).toPandas()

    def _add_brand_logic(self, inputDF):
        return BrandLogic.add_brand_indicator(inputDF)

    def _input_df_cleanup(self, inputDF):
        if "transdate" not in inputDF.columns:
            outputDF = (inputDF
                        .withColumnRenamed("{}".format(config.TRANS_DATE),
                                           "transdate")
                        .withColumn("transdate", F.date_format('transdate',
                                                               "yyyy-MM-dd"))
                        )
        else:
            outputDF = inputDF

        if self.by_brand:  # only keeping brand's transactions
            outputDF = self._add_brand_logic(outputDF)
            outputDF = (outputDF.filter("brand_indicator = 1")
                                .select([col for col in outputDF.columns
                                         if col not in ["brand_indicator"]])
                        )

        null_counts = self._count_columns(outputDF)
        if not null_counts.eq(null_counts.iloc[0, 0], axis=0).all(1)[0]:
            outputDF = outputDF.dropna(how='any')

        return outputDF

    def _get_monetary_t(self, inputDF):
        outputDF = inputDF.withColumn('monetary_t',
                                      F.round(inputDF.quantity
                                              * inputDF.amount, 2))
        return outputDF

    def _get_recency_t(self, inputDF):
        w = Window.partitionBy("{}".format(config.CUST_ID)).orderBy("transdate")
        outputDF = inputDF.withColumn("transdate_lagged",
                                      F.lag("transdate", 1).over(w))
        outputDF = outputDF.withColumn("recency_t",
                                       F.datediff("transdate",
                                                  "transdate_lagged"))
        outputDF = outputDF.dropna(how='any')

        return outputDF

    def build_rfm(self, inputDF, gby_lst):
        df = self._input_df_cleanup(inputDF).cache()
        df.count()
        df = self._get_monetary_t(df).cache()
        df.count()
        df = self._get_recency_t(df).cache()
        df.count()
        recency = df.groupBy(*gby_lst).agg(F.avg('recency_t').alias('recency'))
        frequency = (df.groupBy(*gby_lst, 'orderid').count()
                       .groupBy(*gby_lst)
                       .agg(F.count("*").alias("frequency"))
                     )
        monetary = (df.groupBy(*gby_lst).agg(F.round(F.sum('monetary_t'), 2)
                                              .alias('monetary'))
                    )
        rfm = Utils.join_df(recency, frequency, gby_lst, how='inner')
        rfm = Utils.join_df(rfm, monetary, gby_lst, how='inner')

        return rfm

    def _describe_pd(self, df, columns, style=1):
        '''
        Function to union the basic stats results and deciles
        Arguments
            - df: the input dataframe
            - columns: the cloumn name list of the numerical variable
            - style: the display style
        Returns
            - The numerical describe info. of the input dataframe
        '''

        if style == 1:
            percentiles = [25, 50, 75]
        else:
            percentiles = np.array(range(0, 110, 10))

        percs = np.transpose([np.percentile(df.select(x).collect(),
                                            percentiles) for x in columns])
        percs = pd.DataFrame(percs, columns=columns)
        percs['summary'] = [str(p) + '%' for p in percentiles]

        spark_describe = df.describe().toPandas()
        new_df = pd.concat([spark_describe, percs], ignore_index=True)
        new_df = new_df.round(2)

        return new_df[['summary'] + columns]

    def _get_quartiles(self, rfm: DataFrame, col):
        percentiles = [25, 50, 75]
        return list(np.percentile(rfm.select(col).collect(), percentiles))

    def RScore(self, x, t25, t50, t75):
        if x <= t25:
            return 1
        elif x <= t50:
            return 2
        elif x <= t75:
            return 3
        else:
            return 4

    def FMScore(self, x, t25, t50, t75):
        if x <= t25:
            return 4
        elif x <= t50:
            return 3
        elif x <= t75:
            return 2
        else:
            return 1

    def add_rfm_segment(self, rfm: DataFrame):

        r_thresh = self._get_quartiles(rfm, "recency")
        f_thresh = self._get_quartiles(rfm, "frequency")
        m_thresh = self._get_quartiles(rfm, "monetary")

        R_udf = F.udf(lambda x: self.RScore(x, *r_thresh), StringType())
        F_udf = F.udf(lambda x: self.FMScore(x, *f_thresh), StringType())
        M_udf = F.udf(lambda x: self.FMScore(x, *m_thresh), StringType())

        rfm_seg = rfm.withColumn("r_seg", R_udf("recency"))
        rfm_seg = rfm_seg.withColumn("f_seg", F_udf("frequency"))
        rfm_seg = rfm_seg.withColumn("m_seg", M_udf("monetary"))
        rfm_seg = rfm_seg.withColumn('RFMScore',
                                     F.concat(F.col('r_seg'),
                                              F.col('f_seg'),
                                              F.col('m_seg')))
        return rfm_seg

    def _remove_fraud(self, rfm_seg: DataFrame):
        if False:    # TODO: need to find a better way for fraud cases
            mean_freq = (rfm_seg.filter("RFMScore like '1%'")
                         .select(F.avg("frequency").alias("avg_freq"))
                         .collect()[0]["avg_freq"])
            stdv_freq = (rfm_seg.filter("RFMScore like '1%'")
                         .select(F.stddev("frequency").alias("std_freq"))
                         .collect()[0]["std_freq"])

            fraud_thresh = mean_freq + 2*stdv_freq

        fraud_thresh = self._get_quartiles(rfm_seg.select("frequency"),
                                           "frequency")[-1]
        rfm_seg = (rfm_seg.filter(~((F.col("recency") == 0)
                                    & (F.col("frequency") > fraud_thresh)))
                   )
        return rfm_seg

    def get_rfm_segment(self, inputDF, gby_lst, remove_fraud=True):
        rfm = self.build_rfm(inputDF, gby_lst).cache()
        rfm.count()
        rfm_seg = self.add_rfm_segment(rfm).cache()
        rfm_seg.cache()
        if remove_fraud:
            rfm_ = self._remove_fraud(rfm_seg).cache()
            rfm_.count()
            rfm_ = rfm_.select(*gby_lst, "recency", "frequency", "monetary")
            rfm_seg = self.add_rfm_segment(rfm_)
            rfm_seg.cache()

        drop_lst = ["recency", "frequency", "monetary",
                    "r_seg", "f_seg", "m_seg"]
        rfm_seg = rfm_seg.select([col for col in rfm_seg.columns
                                 if col not in drop_lst])
        return rfm_seg
