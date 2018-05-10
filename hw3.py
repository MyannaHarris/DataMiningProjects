#!/usr/bin/python
# CPSC 462 Data Mining
# HW3
# Kristina Spring, Myanna Harris
# Oct 3, 2016
# Class prediction algorithms

import matplotlib.pyplot as pyplot
import numpy
import math
import random
import tabulate

#function from hw1
def replaceMissingWMeaningfulAvg(origFile, newFile, attr):
    file = open(origFile, "r")
    rows = [r.split(",") for r in filter(None, file.read().split("\n"))]
    file.close()
    out = []
    keys = attr.keys()
    for i in range(0,len(keys)):
        key = keys[i]
        val = attr[key]
        for r in rows:
            if r[key] == "NA":
                meaningfulList = [float(r2[key]) for r2 in rows if (r2[key]!="NA"and r[6] == r2[6])]
                r[key] = str(getAvg(meaningfulList))
    for r in rows:
        out.append(",".join(r))
    file = open(newFile, "w")
    file.write("\n".join(out))
    file.close()

#function from hw1
def getAvg(list):
    total = 0.0
    for x in list:
        total += float(x)
    return total/(len(list)*1.0)

#function from hw2
def csvToTable(fileName):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\n"))
    file.close()
    table = []
    for row in rows:
        splitRow = row.split(",")
        table.append(splitRow)
    return table

#function from hw2
def getCol(table, index):
    return [ row[index] for row in table ]

#function from hw2
# Returns (m(slope), b(y-intercept), correlation coefficient, covariance)
def leastSquares(table, xIdx, yIdx):
    xCol = getCol(table, xIdx)
    xAvg = getAvg(xCol)
    yAvg = getAvg(getCol(table, yIdx))
    m = sum([(float(r[xIdx])-xAvg)*(float(r[yIdx])-yAvg) for r in table])/ ((sum([(float(r[xIdx])-xAvg)*(float(r[xIdx])-xAvg) for r in table]))*1.0)
    b = yAvg - (xAvg * m)
    numerator = float(sum([(float(r[xIdx])-xAvg) * (float(r[yIdx])-yAvg) for r in table]))
    corrDenominator = float( math.sqrt( sum([(float(r[xIdx])-xAvg) * (float(r[xIdx])-xAvg) for r in table]) * sum([(float(r[yIdx])-yAvg) * (float(r[yIdx])-yAvg) for r in table]) ) )
    corr = numerator / corrDenominator
    cov = numerator / (len(xCol)*1.0)
    return (m, b, corr, cov)

def getDOERanking(val):
    if val <= 13:
        return 1
    elif val <= 14:
        return 2
    elif val <= 16:
        return 3
    elif val <= 19:
        return 4
    elif val <= 23:
        return 5
    elif val <= 26:
        return 6
    elif val <= 30:
        return 7
    elif val <= 36:
        return 8
    elif val <= 44:
        return 9
    else:
        return 10

# compares least square regression prediction to actual
def leastSquaresComparison(row, m, b, isPrinting):
    if isPrinting:
        print("instance: "+", ".join(row))
    prediction = getDOERanking(m*float(row[4])+b)
    actual = getDOERanking(float(row[0]))
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

# chooses knn result
def chooseNearest(results):
    return (2*results[0][1]+results[1][1]+results[2][1]+0.5*results[3][1]+0.5*results[4][1])/5.0

# checks if row is in k nn
def checkResults(results,distance,val):
    if distance < results[-1][0]:
        place = len(results)-1
        while distance < results[place-1][0] and place > 0:
            place -= 1
        results.insert(place,[distance,float(val)])
        results.pop()

# normalized distance
def findNormalizedDistance(col,val1,val2):
    floatCol = [float(x) for x in col]
    maxVal = max(floatCol)
    minVal = min(floatCol)
    normalizedVal1 = (val1-minVal)/float(maxVal-minVal)
    normalizedVal2 = (val2-minVal)/float(maxVal-minVal)
    return normalizedVal1 - normalizedVal2

# table to compare to
# instance to compare
# columns to use to calculate distance
# 1 = printing predictiona dn actual, 0 = not printing
def fiveNearestNeighbors(table,instance,cols,isPrinting):
    if isPrinting:
        print("instance: "+", ".join(instance))
    results = [[300000,1]]*5
    for row in table:
        if row != instance:
            distanceSquared = 0
            for col in cols:
                normalizedDistance = findNormalizedDistance(getCol(table,col),float(row[col]),float(instance[col]))
                distanceSquared += pow(normalizedDistance,2)
            distance = math.sqrt(distanceSquared)
            checkResults(results,distance,row[0])
    prediction = getDOERanking(chooseNearest(results))
    actual = getDOERanking(float(instance[0]))
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

# gets random subsamples  
def randomSubsample(table):
    tableTrain = []
    tableTest = []
    trainIdx = random.sample(range(0, len(table)), len(table)*2/3)
    for i in range(0, len(table)):
        if i in trainIdx:
            tableTrain.append(table[i])
        else:
            tableTest.append(table[i])
    return (tableTrain, tableTest)

# gets statistically accurate sub samples
def stratifiedSubsamples(table,k):
    subSamples = [[] for i in range(0, k)]
    classList = [[] for i in range(0, k)]
    for row in table:
        rank = getDOERanking(row[0])
        classList[rank-1].append(row)
    for classTable in classList:
        num = len(classTable) / k
        extra = num % k
        start = 0
        for x in range(0, k):
            end = start + num
            if x == k-1:
                end += extra
            for z in range(start, end):
                subSamples[x].append(classTable[z])
            start += num
    return subSamples

# accuracy
# correct hits divided by total hits
def calcAccuracy(confusionMatrix):
    trues = 0
    total = 0
    for i in range(0, 10):
        trues += confusionMatrix[i][i]
        total += sum(confusionMatrix[i])
    return trues / (total * 1.0)

# error rate
def calcErrorRate(accuracy):
    return 1.0 - accuracy

def main():
    # continuous attributes
    contDict = { 0:"MPG",
		 2:"displacement",
		 3:"horsepower",
		 4:"weight",
		 5:"acceleration",
		 9:"MSRP" }
    # replace missing attributes in table with meaningful averages
    # save new table to new file
    replaceMissingWMeaningfulAvg("auto-data.txt", "auto-data-meaningfulavg.txt", contDict)
    # Load newly made table
    table = csvToTable("auto-data-meaningfulavg.txt")

    print("===========================================")
    print("STEP 1: Linear Regression MPG Classifier")
    print("===========================================")
    linearRegressionTuple = leastSquares(table,4,0)
    rand = random.randint(0,len(table)-1)
    leastSquaresComparison(table[rand],linearRegressionTuple[0],linearRegressionTuple[1],1)
    rand = random.randint(0,len(table)-1)
    leastSquaresComparison(table[rand],linearRegressionTuple[0],linearRegressionTuple[1],1)
    rand = random.randint(0,len(table)-1)
    leastSquaresComparison(table[rand],linearRegressionTuple[0],linearRegressionTuple[1],1)
    rand = random.randint(0,len(table)-1)
    leastSquaresComparison(table[rand],linearRegressionTuple[0],linearRegressionTuple[1],1)
    rand = random.randint(0,len(table)-1)
    leastSquaresComparison(table[rand],linearRegressionTuple[0],linearRegressionTuple[1],1)

    print("===========================================")
    print("STEP 2: k=5 Nearest Neighbor MPG Classifier")
    print("===========================================")
    rand = random.randint(0,len(table)-1)
    fiveNearestNeighbors(table,table[rand],[1,4,5],1)
    rand = random.randint(0,len(table)-1)
    fiveNearestNeighbors(table,table[rand],[1,4,5],1)
    rand = random.randint(0,len(table)-1)
    fiveNearestNeighbors(table,table[rand],[1,4,5],1)
    rand = random.randint(0,len(table)-1)
    fiveNearestNeighbors(table,table[rand],[1,4,5],1)
    rand = random.randint(0,len(table)-1)
    fiveNearestNeighbors(table,table[rand],[1,4,5],1)

    print("===========================================")
    print("STEP 3: Predictive Accuracy")
    print("===========================================")

    # RANDOM SUBSAMPLING
    print("   Random Subsample (k=10, 2:1 Train/Test)")

    totalAccuracyLinReg = 0.0
    totalAccuracyKNN = 0.0
    for i in range(0, 10):
        tableTrain, tableTest = randomSubsample(table)

        # LINEAR REGRESSION
        linearRegressionTuple = leastSquares(tableTrain,4,0)

        # accuracy info
        confusionMatrixLinReg = [[0 for p in range(0,10)] for l in range(0,10)]

        # K-NN
        # accuracy info
        confusionMatrixKNN = [[0 for p in range(0,10)] for l in range(0,10)]

        for testRow in tableTest:
            # LINEAR REGRESSION
            prediction, actual = leastSquaresComparison(testRow,linearRegressionTuple[0],linearRegressionTuple[1],0)
            confusionMatrixLinReg[prediction][actual] += 1

            # K-NN
            prediction, actual = fiveNearestNeighbors(tableTrain,testRow,[1,4,5],0)
            confusionMatrixKNN[prediction][actual] += 1

        totalAccuracyLinReg += calcAccuracy(confusionMatrixLinReg)

        totalAccuracyKNN += calcAccuracy(confusionMatrixKNN)

    accuracy = totalAccuracyLinReg/10.0
    print("      Linear Regression: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

    accuracy = totalAccuracyKNN/10.0
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamples(table, 10)

    # LINEAR REGRESSION
    # accuracy info
    confusionMatrixLinReg = [[0 for p in range(0,12)] for l in range(0,10)]

    # K-NN
    # accuracy info
    confusionMatrixKNN = [[0 for p in range(0,12)] for l in range(0,10)]

    totalAccuracyLinReg = 0.0
    totalAccuracyKNN = 0.0
    for i in range(0, 10):
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]

        # LINEAR REGRESSION
        linearRegressionTuple = leastSquares(tableTrain,4,0)


        for testRow in tableTest:
            # LINEAR REGRESSION
            prediction, actual = leastSquaresComparison(testRow,linearRegressionTuple[0],linearRegressionTuple[1],0)
            confusionMatrixLinReg[prediction][actual] += 1

            # K-NN
            prediction, actual = fiveNearestNeighbors(tableTrain,testRow,[1,4,5],0)
            confusionMatrixKNN[prediction][actual] += 1

    accuracy = calcAccuracy(confusionMatrixLinReg)
    print("      Linear Regression: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

    accuracy = calcAccuracy(confusionMatrixKNN)
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

    print("===========================================")
    print("STEP 4: Confusion Matrices")
    print("===========================================")

    for w in range(0, 10):
        trues = confusionMatrixLinReg[w][w]
        total = sum(confusionMatrixLinReg[w])
        confusionMatrixLinReg[w][10] = trues
        if total > 0:
            confusionMatrixLinReg[w][11] = trues / (total*1.0) * 100
        else:
            confusionMatrixLinReg[w][11] = 0
        trues = confusionMatrixKNN[w][w]
        total = sum(confusionMatrixKNN[w])
        confusionMatrixKNN[w][10] = trues
        if total > 0:
            confusionMatrixKNN[w][11] = trues / (total*1.0) * 100
        else:
            confusionMatrixKNN[w][11] = 0
    
    print("Linear Regression (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(confusionMatrixLinReg,headers=["1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")

    print("k Nearest Neighbors (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(confusionMatrixKNN,headers=["1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)


if __name__ == '__main__':
    main()
