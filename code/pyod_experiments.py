import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import time

from pyod.utils.utility import precision_n_scores
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def runSingleConfiguration(data, features, groundTruth, algorithmClass, alogrithmParams):
    X = _processInput(data, features)
    
    clf = algorithmClass(**alogrithmParams)

    startTime = time()

    clf.fit(X)
    scores_pred = clf.decision_function(X)
    y_pred = clf.predict(X)

    endTime = time()

    metrics = _calculateMetrics(scores_pred, y_pred, groundTruth)

    configStr = ""
    featureString = ""
    for k, v in alogrithmParams.items():
        configStr += str(k) + " = " + str(v) + "; "
    for f in features:
        featureString += f + ", "
    metrics["config"] = configStr[:-2]
    metrics["features"] = featureString[:-2]  
    metrics["execution_time"] = endTime - startTime
    metrics["scores"] = scores_pred
    metrics["labels"] = y_pred

    return metrics
    
def runVariableConfiguration(data, features, groundTruth, algorithmClass, alogrithmParams, variableParamName, variableMin, variableMax, variableSteps=1, variableType="int"):
    X = _processInput(data, features)

    results = { "value": [], "avg_precision": [], "fp": [], "fn": [] }


    float_range = np.linspace(variableMin, variableMax, variableSteps).tolist()

    for var in float_range:
        if variableType is "int":
            var = int(var)
        elif variableType is "float":
            var = round(var, 2)

        alogrithmParams[variableParamName] = var

        clf = algorithmClass(**alogrithmParams)

        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

        metrics = _calculateMetrics(scores_pred, y_pred, groundTruth)

        results["value"].append(var)
        results["avg_precision"].append(metrics["avg_precision"])
        results["fp"].append(metrics["fp"])
        results["fn"].append(metrics["fn"])

    df = pd.DataFrame(results)
    df = df.set_index(df["value"])
    
    return df

def runVariableCategoricalConfiguration(data, features, groundTruth, algorithmClass, alogrithmParams, variableParamName, categoryValues):
    X = _processInput(data, features)

    results = { "value": [], "avg_precision": [], "fp": [], "fn": [] }


    for var in categoryValues:
        alogrithmParams[variableParamName] = var

        clf = algorithmClass(**alogrithmParams)

        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
        metrics = _calculateMetrics(scores_pred, y_pred, groundTruth)

        results["value"].append(var)
        results["avg_precision"].append(metrics["avg_precision"])
        results["fp"].append(metrics["fp"])
        results["fn"].append(metrics["fn"])

    df = pd.DataFrame(results)
    df = df.set_index(df["value"])
    
    return df

def _calculateMetrics(predictionScores, predictionLabels, groundTruth):
    labels = predictionLabels.tolist()
    
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    
    for i in range(len(groundTruth)):
        prediction = labels[i]
        actual = groundTruth[i]

        if prediction > 0 and actual == 0: fp += 1
        elif prediction == 0 and actual > 0: fn += 1
        elif prediction == 0 and actual == 0: tn += 1
        elif prediction > 0 and actual > 0: tp += 1

    return {
        "p_at_n": precision_n_scores(groundTruth, predictionScores.tolist()),
        "fp": fp,
        "fn": fn,
        "auc": roc_auc_score(groundTruth, predictionScores),
        "avg_precision": average_precision_score(groundTruth, predictionScores)
    }

def _processInput(df, features):
    scaler = MinMaxScaler(feature_range=(0, 1))

    df[features] = scaler.fit_transform(df[features])

    xf = []
    for featureName in features:
        xf.append(df[featureName].values.reshape(-1, 1))

    x = np.concatenate(tuple(xf), axis=1)
    return x