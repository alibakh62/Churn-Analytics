import numpy as np
import pandas as pd
from churn import config
from datetime import datetime
from sklearn.metrics import confusion_matrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def get_feature_target(inputDF, feature_columns):
    va = VectorAssembler(inputCols=feature_columns, outputCol='features')
    outputDF = va.transform(inputDF)
    outputDF = outputDF.select(['features', 'segment'])
    return outputDF


def train_test_split(inputDF, test_ratio=0.2):
    return inputDF.randomSplit([1-test_ratio, test_ratio], seed=23)


def logistic_classifier(trainDF, save_model=False):
    lr = LogisticRegression(featuresCol='features', labelCol='segment')
    lrModel = lr.fit(trainDF)
    if save_model:
        version = datetime.strftime(datetime.now(), '%Y%m%d')
        lrModel.save("{0}lrMulti_{1}".format(config.MODELSPATH, version))
    return lrModel


def one_vs_rest(trainDF, save_model=False):
    lr = LogisticRegression(featuresCol='features', labelCol='segment')
    ovr = OneVsRest(featuresCol='features', labelCol='segment', classifier=lr)
    ovrModel = ovr.fit(trainDF)
    if save_model:
        version = datetime.strftime(datetime.now(), '%Y%m%d')
        ovrModel.save("{0}ovrMulti_{1}".format(config.MODELSPATH, version))
    return ovrModel


def evaluate(testDF, model):
    predDF = model.transform(testDF)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy",
                                                  labelCol="segment",
                                                  predictionCol="prediction")
    accuracy = evaluator.evaluate(predDF)
    return accuracy


def get_confusion_matrix(testDF, model, labels, normalize=True):
    predDF = model.transform(testDF)
    y_true = predDF.select("segment")
    y_true = y_true.toPandas()
    y_pred = predDF.select("prediction")
    y_pred = y_pred.toPandas()
    cnf_matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        cnf_matrix = (cnf_matrix.astype('float') /
                      cnf_matrix.sum(axis=1)[:, np.newaxis])
    cnf_matrix = pd.DataFrame(cnf_matrix)
    cnf_matrix.columns = labels
    cnf_matrix.index = labels
    return cnf_matrix
