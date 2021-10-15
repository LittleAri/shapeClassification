from statistics import mode
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from copy import deepcopy
import random


def modeValue(values, valuesOrder=[]):
    try:
        # Compute mode of list using the mode function from the statistics library.
        # This function can only output a mode if there is a unique mode.
        md = mode(values)
    except:
        # Find multiple mode values.
        counts = [values.count(i) for i in np.unique(values)]
        modes = np.unique(values)[np.where(np.array(counts) == max(counts))[0]]
        # Now, we either choose the top mode based on some rank, or simply return all mode values:
        if len(valuesOrder) > 0:
            # Choose the mode value that has the lowest rank in the order (e.g. shortest distances)
            k = np.argmin([np.where(np.array(valuesOrder) == p)[0][0] for p in modes])
            md = modes[k]
        else:
            md = modes
    return md


def KNN_distMat(
    distanceMatrix,
    metaData,
    trainingNames,
    testingNames,
    classType=str,
    neighbours=[5],
    f1Average="weighted",
    inMeta=True,
):
    # This function runs a KNN classifier on a training and testing set.
    # K is equal to the neighbours variable.
    # If more than one value is passed in the neighbours variable, then
    # we find the KNN F1 scores using all the neighbours in the list,
    # and output the KNN results that obtained the highest F1 score.
    f1Score = 0
    topNeighbour = neighbours[0]

    distanceMatrix = distanceMatrix.set_index("Name")

    for neigh in neighbours:

        # This will be updated during the Knn algorithm to contain the list of all the pots and their predicted sample.
        # To start with, it only contains the details from the training sample.
        classDetails = {}
        for name in trainingNames:
            classDetails.update(
                {name: {"Class": list(metaData[metaData["Name"] == name]["Class"])[0]}}
            )

        for testName in testingNames:
            dists = []
            # Find the distances between current test value and all the training set.
            for trainName in trainingNames:
                d = distanceMatrix[testName][trainName]
                dists.append(d)
            # Sort the distances from smallest to largest.
            topRank = np.argsort(dists)[:neigh]
            topClasses = []
            for ind in topRank:
                topClasses.append(classDetails[trainingNames[ind]]["Class"])
            # If there are multiple choices for the mode, we pick the one with the smallest distance.
            shapeclass = modeValue(topClasses, topClasses)
            classDetails.update({testName: {"Class": shapeclass}})

        if inMeta == True:
            # If the test set is included in the meta data, then we can compute F1 scores
            # by using the ground truth values in the meta data.
            actual = []
            predicted = []
            for testName in testingNames:
                pred = classType(classDetails[testName]["Class"])
                act = classType(
                    list(metaData[metaData["Name"] == testName]["Class"])[0]
                )
                actual.append(act)
                predicted.append(pred)

            f1 = f1_score(actual, predicted, average=f1Average)

            # If only one neighbour was selected in the neighbour variable, then we output
            # the predicted and actual values from that k. Else, we output the results
            # that obtained the highest F1 score.
            if len(neighbours) > 1:
                if f1 >= f1Score:
                    f1Score = deepcopy(f1)
                    topNeighbour = deepcopy(neigh)
                    predictedClass = deepcopy(predicted)
                    actualClass = deepcopy(actual)
                    results = pd.DataFrame(
                        [testingNames, predictedClass, actualClass]
                    ).T
                    results = results.rename(
                        columns={0: "Name", 1: "predictedClass", 2: "actualClass"}
                    )
            else:
                f1Score = deepcopy(f1)
                predictedClass = deepcopy(predicted)
                actualClass = deepcopy(actual)
                results = pd.DataFrame([testingNames, predictedClass, actualClass]).T
                results = results.rename(
                    columns={0: "Name", 1: "predictedClass", 2: "actualClass"}
                )
        else:
            # If the meta data doesn't include the test set, then we output the results
            # (predicted classes) from the last neighbour and the F1 score is set to "N/A".
            predictedClass = []
            for testName in testingNames:
                pred = classType(classDetails[testName]["Class"])
                predictedClass.append(pred)
            f1Score = "N/A"
            results = pd.DataFrame([testingNames, predictedClass]).T
            results = results.rename(columns={0: "Name", 1: "predictedClass"})

    return results, f1Score, topNeighbour


def KNN_Bootstrapping(
    table,
    all_dists,
    classes,
    kNeighbours=np.int_(np.linspace(3, 12, 10)),
    nBootstrapping=50,
    classType=str,
    trainingSetSize=12,
    trainingProportion=0,
):

    all_scores = np.zeros((1, nBootstrapping))
    top_neighs = np.zeros((1, nBootstrapping))
    all_classification = []

    top_score = 0
    top_results = []
    top_trainingTesting = []

    all_training = []

    for nb in range(0, nBootstrapping):

        # 1) Create training and test sets:
        train_names = []
        test_names = []

        for n, sp in enumerate(classes):
            inds = list(table[table["Class"] == sp]["Index"])
            if trainingProportion != 0:
                trainingSetSize = int(len(inds) * trainingProportion)
            randinds = random.sample(inds, trainingSetSize)
            testnames = list(table["Name"][list(np.setdiff1d(inds, randinds))])
            trainnames = list(table["Name"][randinds])
            train_names.extend(trainnames)
            test_names.extend(testnames)
        all_training.append(train_names)

        results, f1Score, topNeighbour = KNN_distMat(
            all_dists,
            table,
            train_names,
            test_names,
            classType=classType,
            neighbours=kNeighbours,
            f1Average="weighted",
            inMeta=True,
        )

        all_scores[0, nb] = f1Score
        top_neighs[0, nb] = topNeighbour
        all_classification.append(results)

        if f1Score >= top_score:
            top_score = deepcopy(f1Score)
            top_results = deepcopy(results)
            top_trainingTesting = deepcopy([train_names, test_names])

    all_results = {}
    all_results["topF1Score"] = top_score
    all_results["topClassification"] = top_results
    all_results["topTrainingTesting"] = top_trainingTesting
    all_results["allF1Scores"] = all_scores
    all_results["allTopNeighbours"] = top_neighs
    all_results["allTrainingSamples"] = all_training
    all_results["allClassificationResults"] = all_classification

    return all_results
