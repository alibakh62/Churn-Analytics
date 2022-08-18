from collections import Counter
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.session import SparkSession
from pyspark.sql import DataFrame
from utils import Utils
from churn import config


class BrandLogic:

    @staticmethod
    def load_brand_input_data():
        spark = SparkSession.builder.getOrCreate()
        try:
            df = spark.sql("select * from {0}.{1}".format(config.DATABASE,
                                                          config.POS_TRANS_DETAIL))
        except Exception:
            print("table doesn't exist, creating it ...")
            # some initial cleaning
            raw_data = spark.sql("select * from {}".format(config.POS_RAW_DATA))
            raw_data = raw_data.toDF(*[c.lower() for c in raw_data.columns])
            try:
                raw_data = raw_data.withColumn("month",
                                               F.substring("daydate", 6, 2))
            except KeyError:
                print("daydate column doesn't exist!")

            # trans data for loyalty program members only
            df0 = (raw_data.filter("trim(lower({})) = 'member'"
                                   .format(config.IS_MEMBER))
                   .withColumn("prodcat_tree",
                               F.concat_ws('_', F.col(config.PSACODE_COL),
                                           F.col(config.CATCODE_COL),
                                           F.col(config.SUBCATCODE_COL)))
					# creating brand category column
                   .withColumn("brand_cat_flag",
                               F.when(F.lower(F.col(config.CATNAME_COL))
                                      .like("%{}%".format(config.BRAND_CAT)) &
                                      F.lower(F.col(config.VENDORNAME_COL))
                                      .like("%{}%".format(config.BRAND_NAME)), 'brand_cat')
                                .when(F.lower(F.col(config.CATNAME_COL))
                                      .like("%{}%".format(config.BRAND_CAT)) &
                                      ~F.lower(F.col(config.VENDORNAME_COL))
                                      .like("%{}%".format(config.BRAND_NAME)), 'nonbrand_cat')
                                .otherwise('other')))
            # getting customer who have bought tobacco/cigarette-related products at some point
            tmp1 = df0.select("{}".format(config.CUST_ID),
                              "{}".format(config.PSANAME_COL),
                              "prodcat_tree", "brand_cat_flag")
            tmp2 = (tmp1.groupBy("{}".format(config.CUST_ID))
                    .agg(F.collect_set(config.PSANAME_COL).alias("psa_set"),
                         F.collect_set("prodcat_tree").alias("prodtree_set"),
                         F.collect_set("brand_cat_flag").alias("brand_cat_set")))
            df1 = (tmp2.select("{}".format(config.CUST_ID),
                               "psa_set",
                               "prodtree_set",
                               "brand_cat_set").filter("lower(cast(psa_set as string)) \
                                                   like '%{0}%' or \
                                                   lower(cast(psa_set as string)) \
                                                   like '%{1}%'"
                                                  .format(config.PSA32,
                                                          config.PSA12)))
            df = Utils.join_df(df0, df1.select("{}".format(config.CUST_ID)),
                               join_keys=[config.CUST_ID], how='inner')
            df.count()
            print("Trans data created, writing it to table...")
            df.createOrReplaceTempView("{}".format(config.POS_TRANS_DETAIL))
            spark.sql("drop table if exists {0}.{1}".format(config.DATABASE,
                                                            config.POS_TRANS_DETAIL))
            spark.sql("create table {0}.{1} as select * from {1}".format(config.DATABASE, config.POS_TRANS_DETAIL))
        return df

    @staticmethod
    def add_brand_indicator(df: DataFrame):
        outdf = (df
                 .withColumn("brand_indicator",
                             F.when(F.lower(F.col("{}".format(config.VENDORNAME_COL)))
                                    .like("%{}%".format(config.BRAND_NAME)), 1)
                             .otherwise(0)
                             )
                 )
        return outdf

    @staticmethod
    def add_brand_segment(df: DataFrame, gby_lst):
        '''
        Adds brand's segment column
        Arguments
        ----------
            - df: customer transaction data
        Returns
        -------
            - segmentDF: spark DataFrame
              contains segment label for each memberid
        '''
        # getting customers who bought psa 12 or 32 at some point
        tmpDF1 = (df
                  .groupby(*gby_lst)
                  .agg(F.collect_set("{}".format(config.PSANAME_COL)).alias("psa_set"),
                       F.collect_set("brand_cat_flag").alias("brand_cat_set"))
                  .filter((F.lower(F.col("psa_set").cast(StringType())).like('%{}%'.format(config.PSA32))) |
                          (F.lower(F.col("psa_set").cast(StringType())).like('%{}%'.format(config.PSA12))))
                  )
        # adding brand column
        tmpDF2 = (tmpDF1
                  .withColumn("brand", F.when(~F.col("brand_cat_set").cast(StringType()).like('%{}%'.format(config.BRAND_NAME)), 'other')
				  .when(~F.col("brand_cat_set").cast(StringType()).like('%nonbrand_cat%'), 'only_brand')
                  .when(~F.col("brand_cat_set").cast(StringType()).like('%brand_cat%'), 'only_nonbrand').otherwise('both_brand_nonbrand')
                              )
                  )
        # adding type column
        tmpDF3 = (tmpDF2
                  .withColumn("type", F.when(~F.lower(F.col("brand_cat_set").cast(StringType())).like('%brand_cat%'), 'brandcat')
                                       .when(F.lower(F.col("psa_set").cast(StringType())).like('%{}%'.format(config.PSA12)), 'brand_nonbrand')
                                       .otherwise('brandcat')
                              )
                  )
        # adding segment column
        segmentDF = (tmpDF3
                     .withColumn("segment", F.when((F.col("brand") == 'only_brand') & (F.col("type") == 'brandcat'), 1)
                                             .when((F.col("brand") == 'only_brand') & (F.col("type") == 'brand_nonbrand'), 4)
                                             .when((F.col("brand") == 'both_brand_nonbrand') & (F.col("type") == 'brandcat'), 2)
                                             .when((F.col("brand") == 'both_brand_nonbrand') & (F.col("type") == 'brand_nonbrand'), 5)
                                             .when((F.col("brand") == 'only_nonbrand') & (F.col("type") == 'brandcat'), 3)
                                             .when((F.col("brand") == 'only_nonbrand') & (F.col("type") == 'brand_nonbrand'), 6)
                                             .otherwise(7)
                                 )
                     .select(*gby_lst, "segment")
                     )
        return segmentDF

    @staticmethod
    def add_brand_features(df: DataFrame, gby_lst):

        outdf = df

        if config.BRAND_BASKET_DESC:
            # amount of smoke-related purchases
            tmpdf = (df.filter("({0} = 12) or ({0} = 32)"
                               .format(config.PSACODE_COL))
                     .groupBy(*gby_lst)
                     .agg(F.avg("{}".format(config.BRAND_TRANS_AMT_COL))
                          .alias("basket_size")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

            # amount of ecig-related purchases
            tmpdf = (df.filter(F.lower(F.col("{}".format(config.CATNAME_COL)))
                               .like("%{}%".format(config.BRAND_CAT)))
                     .groupBy(*gby_lst)
                     .agg(F.avg("{}".format(config.BRAND_TRANS_AMT_COL))
                          .alias("brandcat_basket_size")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

            # number of smoke-related purchases
            tmpdf = (df.filter("({0} = 12) or ({0} = 32)"
                               .format(config.PSACODE_COL))
                     .groupBy(*gby_lst)
                     .agg(F.count("{}".format(config.BRAND_TRANS_ORD_COL))
                          .alias("basket_vol")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

            # number of ecig-related purchases
            tmpdf = (df.filter(F.lower(F.col("{}".format(config.CATNAME_COL)))
                               .like("%{}%".format(config.BRAND_CAT)))
                     .groupBy(*gby_lst)
                     .agg(F.count("{}".format(config.BRAND_TRANS_ORD_COL))
                          .alias("basket_vol")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.BRAND_KIT:
            # number of times bought starter kit
            tmpdf = (df.filter(F.lower(F.col("{}"
                                             .format(config.SUBCATNAME_COL)))
                               .like("%{}%".format(config.BRAND_STARTERKIT))
                               )
                     .groupBy(*gby_lst)
                     .agg(F.count("{}".format(config.BRAND_TRANS_ORD_COL))
                          .alias("starter_kit_count")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

            # number of times bought refill kit
            tmpdf = (df.filter(F.lower(F.col("{}"
                                             .format(config.SUBCATNAME_COL)))
                               .like("%{}%".format(config.BRAND_REFILLKIT))
                               )
                     .groupBy(*gby_lst)
                     .agg(F.count("{}".format(config.BRAND_TRANS_ORD_COL))
                          .alias("refill_kit_count")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.BRAND_CESSATION:
            tmpdf = (df.filter(F.lower(F.col("{}".format(config.CATNAME_COL)))
                               .like("%{}%".format(config.BRAND_CESSATION))
                               )
                     .groupBy(*gby_lst)
                     .agg(F.count("{}".format(config.BRAND_TRANS_ORD_COL))
                          .alias("smoke_cessation_count")
                          )
                     )
            outdf = Utils.join_df(outdf, tmpdf, gby_lst)

        if config.BRAND_MOSTCOMMON:
            @F.udf
            def mode(x):
                return Counter(x).most_common(1)[0][0]

            agg_expr = [mode(F.collect_list(col)).alias(col+"_mostCommon")
                        for col in config.BRAND_MOSTCOMMON_COLS]
            tmpdf = (df.filter("({0} <> 12) and ({0} <> 32)"
                               .format(config.PSACODE_COL))
                     .groupBy(*gby_lst).agg(*agg_expr))

        return outdf