#!/usr/bin/python
# CPSC 462 Data Mining
# HW4
# Kristina Spring, Myanna Harris
# Oct 25, 2016
# Naive Bayes

import math
import random
import tabulate
import numpy

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
    
# turn csv into table skipping title line
def csvToTableSkipTitle(fileName):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\n"))
    file.close()
    table = []
    for i in range(1, len(rows)):
        splitRow = rows[i].split(",")
        table.append(splitRow)
    return table

#function from hw2
def getCol(table, index):
    return [ row[index] for row in table ]

def getNHTSARanking(val):
    if val <= 1999:
        return 1
    elif val <= 2499:
        return 2
    elif val <= 2999:
        return 3
    elif val <= 3499:
        return 4
    else:
        return 5

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

# returns list of unique classifiers 
# and dictionary of each class table
def getUniqueClasses(table, classIdx):
    classLst = []
    classTables = {}
    
    for row in table:
        classCurr = float(row[classIdx])
        if getDOERanking(classCurr) not in classLst:
            classLst.append(getDOERanking(classCurr))
            classTables[getDOERanking(classCurr)] = [row]
        else:
            classTables[getDOERanking(classCurr)].append(row)
            
    return (classLst, classTables)

# table = table of data
# classIdx = index of class column
# classList = list of unique classifiers
# returns dictionary of classes and probabilities
def getClassProbabilities(table, classIdx, classList):
    probDict = {}
    
    for classifier in classList:
        probDict[classifier] = 0
    
    totalRows = len(table)
    for row in table:
        classCurr = float(row[classIdx])
        probDict[getDOERanking(classCurr)] += 1
        
    for key in probDict.keys():
        probDict[key] /= float(totalRows)
    
    return probDict
    
# compares naive bayes prediction to actual
def naiveBayesComparison(row, classIdx, classProbs, classTables, atts, isPrinting):
    if isPrinting:
        print("instance: "+", ".join(row))

    naiveBayesProbs = getNaiveBayesProbs(row, classProbs, classTables, atts)
    
    mostProbRank = 0
    largestProb = 0
    for key in naiveBayesProbs.keys():
        if naiveBayesProbs[key] > largestProb:
            mostProbRank = key
            largestProb = naiveBayesProbs[key]
    
    prediction = mostProbRank
    actual = getDOERanking(float(row[classIdx]))
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

# returns dictionary of classes
# and their probabilities for the row
def getNaiveBayesProbs(row, classProbs, classTables, atts):
    naiveBayesProbs = {}
  
    for key in classTables.keys():
        naiveBayesProbs[key] = 1
        
        totalRows = len(classTables[key])
        attDict = {}  
        for classRow in classTables[key]:
            for att in atts:
                if att not in attDict.keys():
                    attDict[att] = 0
                if row[att] == classRow[att]:
                    attDict[att] += 1
                if att == 4:
                    rowAtt = getNHTSARanking(float(row[att]))
                    classRowAtt = getNHTSARanking(float(classRow[att]))
                    if rowAtt == classRowAtt:
                        attDict[att] += 1
        for att in atts:
            attDict[att] /= float(totalRows)
            naiveBayesProbs[key] *= attDict[att]
        
        naiveBayesProbs[key] *= classProbs[key]
        
    return naiveBayesProbs

# compares naive bayes prediction to actual with continuous attributes
def naiveBayesContinComparison(row, classIdx, classProbs, classTables, atts, continAtts, isPrinting):
    if isPrinting:
        print("instance: "+", ".join(row))
    
    naiveBayesProbs = getNaiveBayesProbsWithContinuous(row, classProbs, classTables, atts, continAtts)
    
    mostProbRank = 0
    largestProb = 0
    for key in naiveBayesProbs.keys():
        if naiveBayesProbs[key] > largestProb:
            mostProbRank = key
            largestProb = naiveBayesProbs[key]
    
    prediction = mostProbRank
    actual = getDOERanking(float(row[classIdx]))
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

# returns dictionary of classes
# and their probabilities for the row
def getNaiveBayesProbsWithContinuous(row, classProbs, classTables, atts, continAtts):
    naiveBayesProbs = {}
    
    for key in classTables.keys():
        naiveBayesProbs[key] = 1
        means = [0 for i in range(0, len(continAtts))]
        sdevs = [0 for i in range(0, len(continAtts))]
        
        for i in range(0, len(continAtts)):
            index = continAtts[i]
            numList = [float(classRow[index]) for classRow in classTables[key]]
            numList = numpy.array(numList)
            sdevs[i] = numpy.std(numList)
        
        totalRows = len(classTables[key])
        attDict = {}  
        for classRow in classTables[key]:
            for att in atts:
                if att not in attDict.keys():
                    attDict[att] = 0
                if row[att] == classRow[att]:
                    attDict[att] += 1
            for i in range(0, len(continAtts)):
                index = continAtts[i]
                means[i] += float(classRow[index])
                    
        for att in atts:
            attDict[att] /= float(totalRows)
            naiveBayesProbs[key] *= attDict[att]
            
        for i in range(0, len(continAtts)):
            index = continAtts[i]
            mean= means[i]/float(totalRows)
            sdev = sdevs[i]
            naiveBayesProbs[key] *= gaussian(float(row[index]), mean, sdev)
        
        naiveBayesProbs[key] *= classProbs[key]
    
    return naiveBayesProbs

def gaussian(x, mean, sdev):
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second

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
        extra = len(classTable) % k
        start = 0
        for x in range(0, k):
            end = start + num
            if x == k-1:
                end += extra
            for z in range(start, end):
                subSamples[x].append(classTable[z])
            start += num
    return subSamples
    
# gets statistically accurate sub samples
def stratifiedSubsamplesGen(table,k, classIdx):
    subSamples = [[] for i in range(0, k)]
    classList = {}
    for row in table:
        classRank = row[classIdx]
        if not classList.has_key(classRank):
            classList[classRank] = []
        classList[classRank].append(row)
    
    for key in classList.keys():
        classTable = classList[key]
        num = len(classTable) / k
        extra = len(classTable) % k
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
def calcAccuracy(confusionMatrix, size):
    trues = 0
    total = 0
    for i in range(0, size):
        trues += confusionMatrix[i][i]
        total += sum(confusionMatrix[i])
    return trues / (total * 1.0)

# error rate
def calcErrorRate(accuracy):
    return 1.0 - accuracy 

# Titanic general all categorical
# returns list of unique classifiers 
# and dictionary of each class table
def getUniqueClassesGen(table, classIdx):
    classLst = []
    classTables = {}
    
    for row in table:
        classCurr = row[classIdx]
        if classCurr not in classLst:
            classLst.append(classCurr)
            classTables[classCurr] = [row]
        else:
            classTables[classCurr].append(row)
            
    return (classLst, classTables)

# Titanic general all categorical
# table = table of data
# classIdx = index of class column
# classList = list of unique classifiers
# returns dictionary of classes and probabilities
def getClassProbabilitiesGen(table, classIdx, classList):
    probDict = {}
    
    for classifier in classList:
        probDict[classifier] = 0
    
    totalRows = len(table)
    for row in table:
        classCurr = row[classIdx]
        probDict[classCurr] += 1
        
    for key in probDict.keys():
        probDict[key] /= float(totalRows)
    
    return probDict

# Titanic general all categorical
# compares naive bayes prediction to actual
def naiveBayesComparisonGen(row, classIdx, classProbs, classTables, atts, isPrinting):
    if isPrinting:
        print("instance: "+", ".join(row))
    
    naiveBayesProbs = getNaiveBayesProbsGen(row, classProbs, classTables, atts)
    
    mostProbRank = ""
    largestProb = -1
    for key in naiveBayesProbs.keys():
        if naiveBayesProbs[key] > largestProb:
            mostProbRank = key
            largestProb = naiveBayesProbs[key]
    
    prediction = 0
    if mostProbRank == "yes":
        prediction = 1
    else:
        prediction = 0
    actual = 0
    actRank = row[classIdx]
    if actRank == "yes":
        actual = 1
    else:
        actual = 0
    if isPrinting:
        print("class: "+str(mostProbRank)+", actual: "+str(actRank))
    return (prediction,actual)

# Titanic general all categorical
# returns dictionary of classes
# and their probabilities for the row
def getNaiveBayesProbsGen(row, classProbs, classTables, atts):
    naiveBayesProbs = {}
    
    for key in classTables.keys():
        naiveBayesProbs[key] = 1
        
        totalRows = len(classTables[key])
        attDict = {}  
        for classRow in classTables[key]:
            for att in atts:
                if att not in attDict.keys():
                    attDict[att] = 0
                if row[att] == classRow[att]:
                    attDict[att] += 1
        for att in atts:
            attDict[att] /= float(totalRows)
            naiveBayesProbs[key] *= attDict[att]
        
        naiveBayesProbs[key] *= classProbs[key]
    
    return naiveBayesProbs

# Titanic general all categorical
# chooses knn result
def chooseNearest(results):
    numYes = 0
    numNo = 0
    for result in results:
        if result[1] == "yes":
            numYes += 1
        else:
            numNo += 1
    if numYes > numNo:
        return "yes"
    else:
        return "no"

# Titanic general all categorical
# checks if row is in k nn
def checkResults(results,distance,val):
    if distance < results[-1][0]:
        place = len(results)-1
        while distance < results[place-1][0] and place > 0:
            place -= 1
        results.insert(place,[distance,val])
        results.pop()

# Titanic general all categorical
# distance
def findDistance(val1,val2):
    if (val1 == val2):
        return 0
    else:
        return 1

# Titanic general all categorical
# table to compare to
# instance to compare
# columns to use to calculate distance
# 1 = printing predictiona dn actual, 0 = not printing
def fiveNearestNeighbors(table,instance, classIdx, cols,isPrinting):
    if isPrinting:
        print("instance: "+", ".join(instance))
    results = [[300000,""]]*5
    foundSelf = False
    for row in table:
        if row != instance or foundSelf:
            distanceSquared = 0
            for col in cols:
                normalizedDistance = findDistance(row[col],instance[col])
                distanceSquared += pow(normalizedDistance,2)
            distance = math.sqrt(distanceSquared)
            checkResults(results,distance,row[classIdx])
        else:
            foundSelf = True
    
    mostProbRank = chooseNearest(results)
    
    prediction = 0
    if mostProbRank == "yes":
        prediction = 1
    else:
        prediction = 0
    actual = 0
    actRank = instance[classIdx]
    if actRank == "yes":
        actual = 1
    else:
        actual = 0
    if isPrinting:
        print("class: "+mostProbRank+", actual: "+actRank)
    return (prediction,actual)

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
    print("STEP 1 Part 1: Naive Bayes MPG Classifier with Categorical")
    print("===========================================")
    
    # Get all class probabilities
    classList, classTables = getUniqueClasses(table, 0)
    classProbs = getClassProbabilities(table, 0, classList)
    
    #indices of attributes to predict based on
    # cylinders = 1, weight = 4, model = 6
    atts = [1, 4, 6]
    
    rand = random.randint(0,len(table)-1)
    naiveBayesComparison(table[rand], 0, classProbs, classTables, atts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesComparison(table[rand], 0, classProbs, classTables, atts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesComparison(table[rand], 0, classProbs, classTables, atts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesComparison(table[rand], 0, classProbs, classTables, atts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesComparison(table[rand], 0, classProbs, classTables, atts, 1)
    
    print("===========================================")
    print("STEP 1 Part 2: Predictive Accuracy")
    print("===========================================")

    # RANDOM SUBSAMPLING
    print("   Random Subsample (k=10, 2:1 Train/Test)")

    totalAccuracyNB = 0.0
    for i in range(0, 10):
        tableTrain, tableTest = randomSubsample(table)
        
        # NAIVE BAYES
        # Get all class probabilities
        classList, classTables = getUniqueClasses(tableTrain, 0)
        classProbs = getClassProbabilities(tableTrain, 0, classList)

        # NAIVE BAYES
        # accuracy info
        confusionMatrixNB = [[0 for p in range(0,10)] for l in range(0,10)]

        for testRow in tableTest:
            # NAIVE BAYES
            prediction, actual = naiveBayesComparison(testRow, 0, classProbs, classTables, atts, 0)
            confusionMatrixNB[actual-1][prediction-1] += 1

        totalAccuracyNB += calcAccuracy(confusionMatrixNB, 10)

    accuracy = totalAccuracyNB/10.0
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamples(table, 10)

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,12)] for l in range(0,10)]

    for i in range(0, 10):
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]
        
        # NAIVE BAYES
        classList, classTables = getUniqueClasses(tableTrain, 0)
        classProbs = getClassProbabilities(tableTrain, 0, classList)


        for testRow in tableTest:
            # NAIVE BAYES
            prediction, actual = naiveBayesComparison(testRow, 0, classProbs, classTables, atts, 0)
            confusionMatrixNB[actual-1][prediction-1] += 1

    accuracy = calcAccuracy(confusionMatrixNB, 10)
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    print("===========================================")
    print("STEP 1 Part 3: Confusion Matrices")
    print("===========================================")
    
    for w in range(0, 10):
        trues = confusionMatrixNB[w][w]
        total = sum(confusionMatrixNB[w])
        confusionMatrixNB[w][10] = trues
        if total > 0:
            confusionMatrixNB[w][11] = trues / (total*1.0) * 100
        else:
            confusionMatrixNB[w][11] = 0
    
    outputMatrix = [[0 for p in range(0,13)] for l in range(0,10)]
    for i in range(0, 10):
        for k in range(0, 13):
            if k == 0:
                outputMatrix[i][k] = i+1
            else:
                outputMatrix[i][k] = confusionMatrixNB[i][k-1]
    
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["MPG","1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    print("===========================================")
    print("STEP 2 Part 1: Naive Bayes MPG Classifier with Continuous")
    print("===========================================")
    
    # Get all class probabilities
    classList, classTables = getUniqueClasses(table, 0)
    classProbs = getClassProbabilities(table, 0, classList)
    
    #indices of attributes to predict based on
    # cylinders = 1, weight = 4, model = 6
    atts = [1, 6]
    continAtts = [4]
    
    rand = random.randint(0,len(table)-1)
    naiveBayesContinComparison(table[rand], 0, classProbs, classTables, atts, continAtts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesContinComparison(table[rand], 0, classProbs, classTables, atts, continAtts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesContinComparison(table[rand], 0, classProbs, classTables, atts, continAtts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesContinComparison(table[rand], 0, classProbs, classTables, atts, continAtts, 1)
    rand = random.randint(0,len(table)-1)
    naiveBayesContinComparison(table[rand], 0, classProbs, classTables, atts, continAtts, 1)
    
    print("===========================================")
    print("STEP 2 Part 2: Predictive Accuracy")
    print("===========================================")

    # RANDOM SUBSAMPLING
    print("   Random Subsample (k=10, 2:1 Train/Test)")

    totalAccuracyNB = 0.0
    for i in range(0, 10):
        tableTrain, tableTest = randomSubsample(table)
        
        # NAIVE BAYES
        # Get all class probabilities
        classList, classTables = getUniqueClasses(tableTrain, 0)
        classProbs = getClassProbabilities(tableTrain, 0, classList)

        # NAIVE BAYES
        # accuracy info
        confusionMatrixNB = [[0 for p in range(0,10)] for l in range(0,10)]

        for testRow in tableTest:
            # NAIVE BAYES
            prediction, actual = naiveBayesContinComparison(testRow, 0, classProbs, classTables, atts, continAtts, 0)
            confusionMatrixNB[actual-1][prediction-1] += 1

        totalAccuracyNB += calcAccuracy(confusionMatrixNB, 10)

    accuracy = totalAccuracyNB/10.0
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamples(table, 10)

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,12)] for l in range(0,10)]

    for i in range(0, 10):
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]
        
        # NAIVE BAYES
        classList, classTables = getUniqueClasses(tableTrain, 0)
        classProbs = getClassProbabilities(tableTrain, 0, classList)


        for testRow in tableTest:
            # NAIVE BAYES
            prediction, actual = naiveBayesContinComparison(testRow, 0, classProbs, classTables, atts, continAtts, 0)
            confusionMatrixNB[actual-1][prediction-1] += 1

    accuracy = calcAccuracy(confusionMatrixNB, 10)
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    print("===========================================")
    print("STEP 2 Part 3: Confusion Matrices")
    print("===========================================")
    
    for w in range(0, 10):
        trues = confusionMatrixNB[w][w]
        total = sum(confusionMatrixNB[w])
        confusionMatrixNB[w][10] = trues
        if total > 0:
            confusionMatrixNB[w][11] = trues / (total*1.0) * 100
        else:
            confusionMatrixNB[w][11] = 0
    
    outputMatrix = [[0 for p in range(0,13)] for l in range(0,10)]
    for i in range(0, 10):
        for k in range(0, 13):
            if k == 0:
                outputMatrix[i][k] = i+1
            else:
                outputMatrix[i][k] = confusionMatrixNB[i][k-1]
    
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["MPG","1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    print("===========================================")
    print("STEP 3 Part 1: Predictive Accuracy")
    print("===========================================")
    
    # Load titanic table
    table = csvToTableSkipTitle("titanic.txt")
    
    #indices of attributes to predict based on
    # class = 0, age = 1, gender = 2
    atts = [0, 1, 2]
    continAtts = []

    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamplesGen(table, 10, 3)

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,4)] for l in range(0,2)]

    # K-NN
    # accuracy info
    confusionMatrixKNN = [[0 for p in range(0,4)] for l in range(0,2)]

    for i in range(0, 10):
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]

        # NAIVE BAYES
        # Get all class probabilities
        classList, classTables = getUniqueClassesGen(tableTrain, 3)
        classProbs = getClassProbabilitiesGen(tableTrain, 3, classList)


        for testRow in tableTest:
            # NAIVE BAYES
            prediction, actual = naiveBayesComparisonGen(testRow, 3, classProbs, classTables, atts, 0)
            confusionMatrixNB[actual][prediction] += 1

            # K-NN
            prediction, actual = fiveNearestNeighbors(tableTrain,testRow, 3, [0,1,2],0)
            confusionMatrixKNN[actual][prediction] += 1

    accuracy = calcAccuracy(confusionMatrixNB, 2)
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

    accuracy = calcAccuracy(confusionMatrixKNN, 2)
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

    print("===========================================")
    print("STEP 3 Part 2: Confusion Matrices")
    print("===========================================")

    for w in range(0, 2):
        trues = confusionMatrixNB[w][w]
        total = sum(confusionMatrixNB[w])
        confusionMatrixNB[w][2] = trues
        if total > 0:
            confusionMatrixNB[w][3] = trues / (total*1.0) * 100
        else:
            confusionMatrixNB[w][3] = 0
        trues = confusionMatrixKNN[w][w]
        total = sum(confusionMatrixKNN[w])
        confusionMatrixKNN[w][2] = trues
        if total > 0:
            confusionMatrixKNN[w][3] = trues / (total*1.0) * 100
        else:
            confusionMatrixKNN[w][3] = 0
    
    outputMatrix = [[0 for p in range(0,5)] for l in range(0,2)]
    for i in range(0, 2):
        for k in range(0, 5):
            if k == 0:
                if i == 0:
                    outputMatrix[i][k] = "No"
                else:
                    outputMatrix[i][k] = "Yes"
            else:
                outputMatrix[i][k] = confusionMatrixNB[i][k-1]
                
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["Survival","No","Yes", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")

    outputMatrix = [[0 for p in range(0,5)] for l in range(0,2)]
    for i in range(0, 2):
        for k in range(0, 5):
            if k == 0:
                if i == 0:
                    outputMatrix[i][k] = "No"
                else:
                    outputMatrix[i][k] = "Yes"
            else:
                outputMatrix[i][k] = confusionMatrixKNN[i][k-1]

    print("k Nearest Neighbors (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["Survival","No","Yes", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)

if __name__ == '__main__':
    main()
