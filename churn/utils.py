from typing import Iterable
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


class Utils:

    @staticmethod
    def melt_df(df: DataFrame,
                id_vars: Iterable[str],
                value_vars: Iterable[str],
                var_name: str = "variable",
                value_name: str = "value") -> DataFrame:
        """
        Gets a pivot dataframe and melt (unpivot) it.

        Arguments:
          - df : The Dataframe on which the operation will be carried out
          - id_vars : array of columns which will be the index to which the
            values of the columns to which matched to.
          - value_vars: while id_vars help use to find the index of the values,
            this is the actual values will be extracted from these columns
          - var_name: name of the variable column in the resulting DataFrame
          - value_name: name of the value variable in the resulting DataFrame
        Returns:
          - melted Dataframe
        """
        # Create array<struct<variable: str, value: ...>>
        _vars_and_vals = F.array(*(
            F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name))
            for c in value_vars))

        # Add to the DataFrame and explode
        _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))
        cols = id_vars + [F.col("_vars_and_vals")[x].alias(x)
                          for x in [var_name, value_name]]

        return _tmp.select(*cols)

    @staticmethod
    def join_df(leftDF, rightDF, join_keys, how='left_outer', dropna=False):

        if len(join_keys) > 1:
            leftDF = leftDF.withColumn("joinkey", F.concat_ws('', *join_keys))
            rightDF = rightDF.withColumn("joinkey", F.concat_ws('', *join_keys))
            rightDF = rightDF.select([col for col in rightDF.columns
                                      if col not in join_keys])
        else:
            leftDF = leftDF.withColumn("joinkey", F.col(*join_keys))
            rightDF = rightDF.withColumn("joinkey", F.col(*join_keys))

        joinDF = (leftDF.alias('A').join(rightDF.alias('B'),
                                         F.col("A.joinkey") == F.col("B.joinkey"),
                                         how)
                  ).select([F.col('A.'+xx) for xx in leftDF.columns
                            if xx not in ["joinkey"]] +
                           [F.col('B.'+xx) for xx in rightDF.columns
                           if xx not in ["joinkey"]+join_keys])
        if dropna:
            joinDF = joinDF.na.drop('any')

        return joinDF
