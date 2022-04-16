"""
file: adaBoost.py
description: CSCI 630, Lab 3, Wikipedia Language Classification
language: Python
author: Abhishek Shah, as5553
"""

import math
from treeNode import *
from checkFeatures import *
import pickle


def calculateEntropy(inputValue):
    """
    Function to calculate entropy
    """
    if inputValue == 1: return 0
    entropy = (-1) * (inputValue * math.log(inputValue, 2.0) + (1 - inputValue) * math.log((1 - inputValue), 2.0))
    return entropy


def appendFeatures(allSentences):
    """
    Creates features list
    """
    feature1, feature2, feature3, feature4, feature5 = ([] for _ in range(5))

    # Based on the sentences fill the values for the features
    for line in allSentences:
        feature1.append(commonDutchWords(line))
        feature2.append(commonEnglishWords(line))
        feature3.append(englishArticles(line))
        feature4.append(stringVan(line))
        feature5.append(stringDeHet(line))
    features = [feature1, feature2, feature3, feature4, feature5]
    return features


def stumpPrediction(stump, statement, features, index):
    """
    For predicting the stump followed by the result it will be giving
    """
    featureValue = stump.value
    if features[featureValue][index] is False: return stump.right.value
    else: return stump.left.value


def adaDataCollection(exampleFile, hypothesisFile):
    """
    Collection of data for Adaboost and collection of stumps formed
    """

    trainData = open(exampleFile, 'r')
    allData = ''
    for lines in trainData:
        allData += lines

    # Get all the statements
    allStatements = allData.split('|')
    countAllStatements = len(allStatements)
    allWords = allData.split()

    for index in range(countAllStatements):
        if index < 1:
            continue
        allStatements[index] = allStatements[index][:-4]
    allStatements = allStatements[1:]

    # all the results
    languageLabel = []
    index = 0
    for word in allWords:
        if word.startswith('nl|') or word.startswith('en|'):
            languageLabel.insert(index, word[:2])
            index = index + 1

    exampleWeights = [1 / countAllStatements] * countAllStatements

    # total number of hypothesis
    totalDecisionStumps = 70

    features = appendFeatures(allStatements)
    stumpValues = []
    hypoWeights = [1.0] * totalDecisionStumps
    print(hypoWeights)
    indexes = [i for i in range(len(languageLabel))]

    # adaBoost algorithm for training the model
    for hypo in range(totalDecisionStumps):
        rootNode = tree(features, None, languageLabel, indexes, 0, None, None)
        # For every hypothesis index generate a hypothesis to be added
        stumpValue = returnDecisionStump(0, rootNode, features, languageLabel, indexes, exampleWeights)
        error = 0.1
        correct = 0
        incorrect = 0
        total = 0
        for index in range(len(allStatements)):
            # Check for number of examples that do not match with hypothesis output value and update error value
            if stumpPrediction(stumpValue, allStatements[index], features, index) != languageLabel[index]:
                error = error + exampleWeights[index]
                incorrect = incorrect + 1

        for index in range(len(allStatements)):
            # Check for number of examples that do mathc with the hypothesis output value and update weights of examples
            if stumpPrediction(stumpValue, allStatements[index], features, index) == languageLabel[index]:
                exampleWeights[index] = exampleWeights[index] * error / (1 - error)
                correct = correct + 1

        # normalization
        for weight in exampleWeights:
            total = total + weight
        for index in range(len(exampleWeights)):
            exampleWeights[index] = exampleWeights[index] / total

        # Updating  hypothesis weight values
        errorUpdated = (1 - error) / error
        hypoWeights[hypo] = math.log(errorUpdated, 2)
        stumpValues.append(stumpValue)

    # dump the model
    saveModel = open(hypothesisFile, 'wb')
    pickle.dump((stumpValues, hypoWeights), saveModel)


def returnDecisionStump(depth, rootNode, features, languageLabel, indexOfExamples, weights):
    """
    Returns a decision stump
    """
    gain = []
    enResult = 0
    nlResult = 0
    for index in indexOfExamples:
        if languageLabel[index] == 'en':
            enResult = enResult + 1 * weights[index]
        else:
            nlResult = nlResult + 1 * weights[index]

    for featureIndex in range(len(features)):
        enTrue, enFalse, nlTrue, nlFalse = 0, 0, 0, 0
        for index in indexOfExamples:
            if features[featureIndex][index] is True and languageLabel[index] == 'en': enTrue = enTrue + 1 * weights[index]
            elif features[featureIndex][index] is True and languageLabel[index] == 'nl': nlTrue = nlTrue + 1 * weights[index]
            elif features[featureIndex][index] is False and languageLabel[index] == 'en': enFalse = enFalse + 1 * weights[index]
            elif features[featureIndex][index] is False and languageLabel[index] == 'nl': nlFalse = nlFalse + 1 * weights[index]

        allFalse = enFalse + nlFalse
        allTrue = enTrue + nlTrue
        allResult = nlResult + enResult

        if enTrue == 0:
            trueValuePending = 0
            falseValuePending = (allFalse / allResult) * calculateEntropy(enFalse / allFalse)

        elif enFalse == 0:
            falseValuePending = 0
            trueValuePending = (allTrue / allResult) * calculateEntropy(enTrue / allTrue)

        else:
            trueValuePending = (allTrue / allResult) * calculateEntropy(enTrue / allTrue)

            falseValuePending = (allFalse / allResult) * calculateEntropy(enFalse / allFalse)

        informationGain = calculateEntropy(enResult / allResult) - (trueValuePending + falseValuePending)
        gain.append(informationGain)

    maxGainFeature = gain.index(max(gain))
    # make that the root node
    rootNode.value = maxGainFeature

    enTrueMax = 0
    nlTrueMax = 0
    enFalseMax = 0
    nlFalseMax = 0

    for index in range(len(features[maxGainFeature])):
        if features[maxGainFeature][index] is True:
            if languageLabel[index] == 'en':
                enTrueMax = enTrueMax + 1 * weights[index]
            else:
                nlTrueMax = nlTrueMax + 1 * weights[index]
        else:
            if languageLabel[index] == 'en':
                enFalseMax = enFalseMax + 1 * weights[index]
            else:
                nlFalseMax = nlFalseMax + 1 * weights[index]

    rootNode.left = tree(features, None, languageLabel, None, depth + 1, None, None)
    rootNode.right = tree(features, None, languageLabel, None, depth + 1,None, None)

    if enTrueMax < nlTrueMax: rootNode.left.value = 'nl'
    else: rootNode.left.value = 'en'

    if enFalseMax < nlFalseMax: rootNode.right.value = 'nl'
    else: rootNode.right.value = 'en'

    return rootNode


def adaStumpPredict(stump, sentence, features, index):
    """
    Returns prediction based on the hypothesis stump
    """
    featureValue = stump.value
    if features[featureValue][index] is True:
        if stump.left.value == 'en':
            return 1
        else:
            return -1
    else:
        if stump.right.value == 'en':
            return 1
        else:
            return -1


def limit15Words(file):
    """
    The program requires definite sized sentence to be checked
    15 in our case
    """
    testDataFile = open(file)
    line = ""
    allStatements = []
    wordCounter = 0

    # grab 15-word sentence from the testData file
    for line in testDataFile:
        words = line.split()
        for word in words:
            if wordCounter != 14:
                line = line + word + " "
                wordCounter = wordCounter + 1
            else:
                line = line + word
                allStatements.append(line)
                line = ""
                wordCounter = 0

    return line, allStatements


def adaPredict(hypothesis, file):
    """
    Making predictions using the saved adaBoost model
    """
    # Loading model from the file
    loadModel = pickle.load(open(hypothesis, 'rb'))

    line, allStatements = limit15Words(file)

    features = appendFeatures(allStatements)

    hypoWeights = loadModel[1]
    hypoList = loadModel[0]
    statementCounter = 0

    # for every 15-word sentence make a prediction
    for line in allStatements:
        total = 0
        for index in range(len(loadModel[0])):
            total = total + adaStumpPredict(hypoList[index], line, features, statementCounter) * hypoWeights[index]

        if total > 0: print('en')
        else: print('nl')

        statementCounter = statementCounter + 1
