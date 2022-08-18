import pandas as pd
from typing import Iterable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import column_or_1d
from pyspark.sql import DataFrame
from lightgbm import LGBMClassifier
from pyspark.sql.session import SparkSession
from dataloader import DataLoader
import pyspark.sql.functions as F


class Params:

    dummy_cols = ["state_mostCommon", "RFMScore", "psaname_mostCommon",
                  "customer_segment"]
    drop_cols = ["segment_prev", "city_mostCommon", "categoryname_mostCommon"]
    model = LogisticRegression

    @property
    def params(cls):
        return cls.dummy_cols, cls.drop_cols, cls.model

    @params.setter
    def params(cls, dummy_cols, drop_cols, model):

        model_dict = {"LogisticRegression": LogisticRegression,
                      "GradientBoostingClassifier": GradientBoostingClassifier,
                      "LGBMClassifier": LGBMClassifier}

        cls.dummy_cols = dummy_cols
        cls.drop_cols = drop_cols
        cls.model = model_dict[model]


def get_segment_ij_data(inputDF, base, target):
    outputDF = inputDF.filter("segment_prev = {}".format(base))
    outputDF = outputDF.filter("segment in ({0},{1})".format(base, target))
    outputDF = outputDF.withColumn("segment",
                                   F.when(F.col("segment") == base, 0)
                                   .otherwise(1))
    return outputDF


def prep_data(df: DataFrame,
              dummy_cols=Params.dummy_cols,
              drop_cols=Params.drop_cols) -> Iterable[pd.DataFrame]:
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    spark.conf.set("spark.driver.maxResultSize", "10g")
    df = df.toPandas()
    df = df.set_index(["memberid", "chunkid"])
    df_index = df.index
    df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)
    df = df.fillna(0)
    df = df.drop(drop_cols, axis=1, errors='ignore').copy()
    xcols = [col for col in df.columns if col != 'segment']
    ycol = ['segment']
    X = df[xcols].copy()
    y = df[ycol].copy()
    return X, y, df_index


def get_trans_prob(X, y, model=Params.model):
    m = model()
    m.fit(X, column_or_1d(y))
    return m.predict_proba(X)[:, 1]


def create_seg_ij_data(ypred, i, j, index):
    df = pd.DataFrame()
    df["trans_prob"] = ypred
    df["segment_i"] = i
    df["segment_j"] = j
    df.index = index
    df = df[["segment_i", "segment_j", "trans_prob"]]
    df = df.reset_index()
    return df


def main(DF: DataFrame, save=True):
    '''
    creates a trans prob df of segment i->j for each memberid, chunkid
    Arguments
    ---------
        - DF: spark DataFrame object, contains all features for training
    Returns
    -------
        - trans_prob_df: pd.DataFrame
    '''
    spark = SparkSession.builder.getOrCreate()
    trans_prob_df = pd.DataFrame()
    for i in range(1, 8):
        for j in range(1, 8):
            if i != j:
                DF_ = get_segment_ij_data(DF, base=i, target=j)
                X, y, index = prep_data(DF_)
                DF_.unpersist()
                ypred = get_trans_prob(X, y)
                seg_ij = create_seg_ij_data(ypred, i, j, index)
                trans_prob_df = trans_prob_df.append(seg_ij, ignore_index=True)
                del X, y, index, seg_ij

    # converting back to spark df to fill out missing seg i->j with null
    transProbDF = spark.createDataFrame(trans_prob_df)
    transProbDF = DataLoader.join_df(DF.select("memberid", "chunkid"),
                                     transProbDF,
                                     join_keys=["memberid", "chunkid"],
                                     how='left_outer').cache()
    # storing the results in a table
    if save:
        transProbDF.createOrReplaceTempView("transProbDF")
        spark.sql("drop table if exists test.transProbDF")
        spark.sql("create table test.transProbDF as select * from transProbDF")


if __name__ == "__main__":
    main()
