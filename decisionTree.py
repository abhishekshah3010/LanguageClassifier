"""
file: decisionTree.py
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


def limit15Words(file):
    """
    The program requires definite sized sentence to be checked
    15 in our case
    """
    testDataFile = open(file)
    line = ""
    allStatements = []
    wordCounter = 0
    # Grab 15-word sentence from the testData file
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


def dtPredict(hypothesis, file):
    """
    Predicts for the decision tree
    """

    # Load already saved(while training) decision tree model
    loadModel = pickle.load(open(hypothesis, 'rb'))

    _, allStatements = limit15Words(file)

    features = appendFeatures(allStatements)

    statementCounter = 0

    # finding langaugeLabel for each statement
    for _ in allStatements:
        model = loadModel
        while type(model.value) != str:
            value = features[model.value][statementCounter]
            if value:
                model = model.left
            else:
                model = model.right
        print(model.value)
        statementCounter = statementCounter + 1


def totalDifferentValues(resultSet, index):
    """
    Checking for total positive/total negative examples
    """
    value = resultSet[index[0]]
    for i in index:
        if value != resultSet[i]:
            return 1
    return 0


def dtDataCollection(exampleFile, hypothesisFile):
    """
    Collection of data for decision tree
    """
    trainData = open(exampleFile, 'r')
    allData = ""
    for lines in trainData:
        allData = allData + lines

    # get all data in the file
    allStatements = allData.split('|')
    countAllStatements = len(allStatements)
    allWords = allData.split()

    for i in range(countAllStatements):
        if i < 1:
            continue
        allStatements[i] = allStatements[i][:-4]
    allStatements = allStatements[1:]

    # get language label of all sentences
    languageLabel = []
    index = 0
    for word in allWords:
        if word.startswith('nl|') or word.startswith('en|'):
            languageLabel.insert(index, word[:2])
            index = index + 1

    features = appendFeatures(allStatements)
    indexes = [i for i in range(len(languageLabel))]

    # keep track of features visited
    visited = []
    rootNode = tree(features, None, languageLabel, indexes, 0, None, None)

    dtTrain(rootNode, features, visited, languageLabel, indexes, 0, None)

    # Saving/Dumping the hypothesisFile
    saveModel = open(hypothesisFile, 'wb')
    pickle.dump(rootNode, saveModel)


def dtTrain(rootNode, features, visited, languageLabel, indexOfExamples, depth, prevLevelPrediction):
    """
    Selects the best splitting attribute for a given depth.
    Makes a root node for that and connects it the left and right child.
    """
    depth = 1
    # If depth is reached
    if depth == len(features) - 1:
        enLabel = 0
        nlLabel = 0
        for index in indexOfExamples:
            if languageLabel[index] == 'en':
                enLabel = enLabel + 1
            elif languageLabel[index] == 'nl':
                nlLabel = nlLabel + 1
        if enLabel > nlLabel:
            rootNode.value = 'en'
            print("en")
        else:
            rootNode.value = 'nl'
            print("nl")

    # if no statements left
    elif len(indexOfExamples) == 0:
        rootNode.value = prevLevelPrediction
        print(prevLevelPrediction)

    # if only positive or negative sentences left
    elif totalDifferentValues(languageLabel, indexOfExamples) == 0:
        rootNode.value = languageLabel[indexOfExamples[0]]
        print(languageLabel[indexOfExamples[0]])

    elif len(features) == len(visited):
        enLabel = 0
        nlLabel = 0
        for index in indexOfExamples:
            if languageLabel[index] == 'en':
                enLabel = enLabel + 1
            elif languageLabel[index] == 'nl':
                nlLabel = nlLabel + 1
        if enLabel < nlLabel:
            rootNode.value = 'nl'
        else:
            rootNode.value = 'en'

    # search for the best split position
    else:
        gain = []
        enResult = 0
        nlResult = 0

        for i in indexOfExamples:
            if languageLabel[i] == 'en':
                enResult = enResult + 1
            else:
                nlResult = nlResult + 1

        featuresLength = len(features)
        for featureIndex in range(featuresLength):

            # If feature already used for splitting, no gain in splitting again
            if featureIndex in visited:
                gain.append(0)
                continue
            # else see for the best splitting attribute
            else:
                enTrue, enFalse, nlTrue, nlFalse = 0, 0, 0, 0

                # update all true/false values
                for index in indexOfExamples:
                    if features[featureIndex][index] is True and languageLabel[index] == 'en':
                        enTrue = enTrue + 1
                    elif features[featureIndex][index] is True and languageLabel[index] == 'nl':
                        nlTrue = nlTrue + 1
                    elif features[featureIndex][index] is False and languageLabel[index] == 'en':
                        enFalse = enFalse + 1
                    elif features[featureIndex][index] is False and languageLabel[index] == 'nl':
                        nlFalse = nlFalse + 1

                allTrueFalse = (enFalse + nlFalse) / (nlResult + nlResult)
                allTrue = enTrue + nlTrue
                allFalse = enFalse + nlFalse
                allResult = nlResult + enResult
                probTrue = enTrue / (nlTrue + enTrue)
                probFalse = enFalse / (nlFalse + enFalse)

                # if only either positive or negative examples remain, don't split
                if (nlTrue + enTrue == 0) or (enFalse + nlFalse == 0):
                    gain.append(0)
                    continue

                if enTrue == 0:
                    trueValuePending = 0
                    falseValuePending = allTrueFalse * calculateEntropy(probFalse)
                elif enFalse == 0:
                    falseValuePending = 0
                    trueValuePending = (allTrue / allResult) * calculateEntropy(probTrue)
                else:
                    trueValuePending = (allTrue / allResult) * calculateEntropy(probTrue)
                    falseValuePending = (allFalse / allResult) * calculateEntropy(probFalse)

                # find information gain for each attribute
                informationGain = calculateEntropy(enResult / allResult) - (trueValuePending + falseValuePending)
                gain.append(informationGain)

        # if max gain is 0 then return, no more gain possible.
        if max(gain) == 0:
            rootNode.value = prevLevelPrediction
            print(rootNode.value)
            return

        # selecting the maxGain feature
        maxGainFeature = gain.index(max(gain))
        visited.append(maxGainFeature)

        truePortion = []
        falsePortion = []

        # separate out true and false portion for the max gain feature
        for index in indexOfExamples:
            if features[maxGainFeature][index] is True:
                truePortion.append(index)
            else:
                falsePortion.append(index)

        currentLevelPrediction = ''
        if enResult > nlResult:
            currentLevelPrediction = 'en'
        else:
            currentLevelPrediction = 'nl'

        # root node
        rootNode.value = maxGainFeature
        # left child
        rootNode.left = tree(features, None, languageLabel, truePortion, depth + 1,
                             currentLevelPrediction, True)
        # right child
        rootNode.right = tree(features, None, languageLabel, falsePortion, depth + 1,
                              currentLevelPrediction, False)

        # Recurse left and right tree
        dtTrain(rootNode.left, features, visited, languageLabel, truePortion, depth + 1, currentLevelPrediction)
        dtTrain(rootNode.right, features, visited, languageLabel, falsePortion, depth + 1, currentLevelPrediction)

        # deleted last visited
        visited = visited[:-1]
