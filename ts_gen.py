# Simplistic generation of web analytics data by simulating existing time series using Markov chains.
# Different kinds of outliers can be inserted into the generated datasets.
# -----------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
from datetime import timedelta

# Helper class that calculates and holds probabilities used by the generator
class Preprocessor():

    def __init__(self, approximateStateSize=1):
        self.approximateStateSize = approximateStateSize

        self.tsStates = {}
        self.tsTransitionMatrices = {}
    
    # Creates a transition probability matrix for each column of the given dataframe
    # Each time series value range is grouped into the specified number of states to reduce complexity 
    def parse(self, dataset):
              
        # Iterate over columns of the dataset
        for (seriesName, columnData) in dataset.iteritems():

            minValue = columnData.min()
            maxValue = columnData.max() + 1

            numberOfStates = int((maxValue - minValue) / self.approximateStateSize)
            self.tsStates[seriesName] = np.array_split(range(minValue, maxValue), numberOfStates)
                     
            # Initialize a transition matrix of the size numberOfStates x numberOfStates x seasonLength
            self.tsTransitionMatrices[seriesName] = [[0] * numberOfStates for _ in range(numberOfStates)]
            
            # Pairwise iteration over data to count total transitions
            for (i1, val1), (i2, val2) in zip(columnData[:-1].iteritems(), columnData[1:].iteritems()):
                state1 = self._getStateForValue(val1, seriesName)
                state2 = self._getStateForValue(val2, seriesName)

                self.tsTransitionMatrices[seriesName][state1][state2] += 1
                
            
            # Convert totals to probabilities
            for row in self.tsTransitionMatrices[seriesName]:
                totalTransitions = sum(row)
                if totalTransitions > 0:
                    row[:] = [count/totalTransitions for count in row]
    
    # Returns the state for a specified value
    def _getStateForValue(self, val, seriesName):
        stateIndex = 0
        for state in self.tsStates[seriesName]:
            if val in state:
                return stateIndex
            stateIndex += 1
        return -1

    # Returns a random value for a specified state
    def _getValueForState(self, state, seriesName, random):
        return random.choice(self.tsStates[seriesName][state])


# General data generator that can be used with any time series data
class DataGenerator():

    def __init__(self, preprocessor, dataset, startDate="2010-01-01", freq="d"):
        self.preprocessor = preprocessor
        self.preprocessor.parse(dataset)

        self.dataset = dataset
        self.startDate = startDate
        self.freq = freq

    # Generate a dataset of the given size
    def generateSyntheticData(self, periods=365, seed=None, trendForColumns=[], trendSlope=0.25):
        generatedDateRange = pd.date_range(start=self.startDate, periods=periods, freq=self.freq)
        
        resultDict = {"timestamp": generatedDateRange}
        for columnName in self.dataset.columns:
            resultDict[columnName] = self._simulateSeries(columnName, periods, seed, columnName in trendForColumns, trendSlope)
        
        return pd.DataFrame.from_dict(resultDict)


    # Generates an array of n time series values using the transition matrix of a given series
    def _simulateSeries(self, seriesName, n, seed, trend, trendSlope):
        matrix = self.preprocessor.tsTransitionMatrices[seriesName]

        random.seed(seed)

        ts = []
        previousState = random.choice(range(len(matrix[0])))
        
        iteration = 0        
        for day in range(n):
            newState = random.choices(range(len(matrix[previousState])), matrix[previousState])[0]

            newValue = self.preprocessor._getValueForState(newState, seriesName, random)

            if trend:
                newValue += int(round(trendSlope * iteration, 0))

            ts.append(newValue)

            previousState = newState
            iteration += 1

        return ts

# Class for inserting outliers into normal data. Creates a labelled dataset that can be used to evaluate outlier detection algorithms.
class WebAnalyticsOutlierSimulator():

    def __init__(self, normalDataset, seasonLength=7):
        self.normal = normalDataset.copy(deep=True)
        self.result = normalDataset.copy(deep=True)
        self.seasonLength = seasonLength

        originalColumns = self.normal.columns

        # Apply moving average to each column
        for columnName in self.result.columns:
            self.normal[columnName + "_ma"] = self.result[columnName].rolling(self.seasonLength * 2).mean()

        self.result["anomalyClass"] = 0

    # Inserts a level shift of the given length. Can be used to simulate system failures.
    def insertLevelShift(self, columns, absoluteStartIndex=0, length=20, weight=0.5, up=False, seed=None):
        random.seed(seed)

        startIndex = random.randint(absoluteStartIndex, len(self.result) - (length + 1))

        i = 0
        while(True):
            if self.result["anomalyClass"].iloc[startIndex:startIndex + length].sum() == 0:
                break
            if i >= 999:
                raise Exception("No space to insert level shift")
            startIndex = random.randint(absoluteStartIndex, len(self.result) - (length + 1))
            i += 1

        for columnName in columns:
            if up:
                self.result[columnName].iloc[startIndex:startIndex + length] = \
                    self.result[columnName].iloc[startIndex:startIndex + length] + \
                    self.normal[columnName + "_ma"].iloc[startIndex:startIndex + length] * weight
                self.result["anomalyClass"].iloc[startIndex:startIndex + length] = 1
            else:
                self.result[columnName].iloc[startIndex:startIndex + length] = \
                    self.result[columnName].iloc[startIndex:startIndex + length] - \
                    self.normal[columnName + "_ma"].iloc[startIndex:startIndex + length] * weight
                self.result.loc[self.result[columnName] < 0, columnName] = 0
                self.result["anomalyClass"].iloc[startIndex:startIndex + length] = 1

    # Inserts a single peek of the given length. Can be used to simualate DoS attacks.
    def insertPeek(self, columns, absoluteStartIndex=0, length=10, weight=1.0, seed=None):
        random.seed(seed)

        startIndex = random.randint(absoluteStartIndex, len(self.result) - (length + 1))

        i = 0
        while(True):
            if self.result["anomalyClass"].iloc[startIndex:startIndex + length].sum() == 0:
                break
            if i >= 999:
                raise Exception("No space to insert level shift")
            startIndex = random.randint(absoluteStartIndex, len(self.result) - (length + 1))
            i += 1

        for columnName in columns:
            for i in range(length):
                self.result[columnName].iloc[startIndex + i] = \
                    self.result[columnName].iloc[startIndex + i] + \
                    self.normal[columnName + "_ma"].iloc[startIndex + i] * weight

                if(i <= length / 2):
                    weight *= 1.33
                else:
                    weight *= 0.66

        self.result["anomalyClass"].iloc[startIndex:startIndex + length] = 1


    # Inserts single outlying points into the normal data.
    def insertPointOutliers(self, columns, absoluteStartIndex=0, freq=0.05, weight=0.5, seed=None):
        random.seed(seed)

        outliersToInsert = int(len(self.normal) * freq)

        while outliersToInsert > 0:
            indexAvailable = False
            while not indexAvailable:
                index = random.randint(absoluteStartIndex, len(self.normal) - 1)
                indexAvailable = self.result["anomalyClass"].iloc[index] == 0
            
            for columnName in columns:
                self.result[columnName].iloc[index] = self.normal[columnName + "_ma"].iloc[index] * weight * random.randint(3, 5)
            self.result["anomalyClass"].iloc[index] = 1

            outliersToInsert -= 1