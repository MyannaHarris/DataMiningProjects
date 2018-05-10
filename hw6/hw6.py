#!/usr/bin/python
# CPSC 462 Data Mining
# HW6
# Kristina Spring, Myanna Harris
# Nov 22, 2016
# Random Forest

import math
import random
import tabulate
import numpy
from random import shuffle

###### FUNCTIONS FROM PREVIOUS HOMEWORKS ######

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
#function from hw4
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

#function from hw4
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

#function from hw3
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

# HW 5    
def tdidt(instances, atts, domains, classIndex):
    if len(domains) == 0:
        leaves = ['Leaves']
        leaves += partitionStats(instances, classIndex)
        return leaves
    elif len(domains) == 1:
        if sameClass(instances, classIndex):
            total = len(table)
            return ['Leaves',[instances[0][classIndex],total,total,1.0]]
        else:
            partitions = partitionInstances(instances, atts[0], domains[0])
            
            subtree = [atts[0]]
            for key, value in partitions.items():
                valueList = [key]
                if len(value) < 1:
                    valueList.append(tdidt(instances, [], [], classIndex))
                else:
                    valueList.append(tdidt(value, [], [], classIndex))
                subtree.append(valueList)
            return subtree
    else:
        partitionAttIdx = selectAttribute(instances, atts, classIndex)
        partitions = partitionInstances(instances, atts[partitionAttIdx], domains[partitionAttIdx])
        
        subtree = [atts[partitionAttIdx]]
        for key, value in partitions.items():
            valueList = [key]
            if len(value) < 1:
                valueList.append(tdidt(instances, [], [], classIndex))
            else:
                valueList.append(tdidt(value, 
                    [att for att in atts if att != atts[partitionAttIdx]], 
                    [dom for dom in domains if dom != domains[partitionAttIdx]], classIndex))
            subtree.append(valueList)
        return subtree

# HW 5
def sameClass(instances, classIndex):
    classValue = instances[0][classIndex]
    for instance in instances:
        if instance[classIndex] == classValue:
            return False
    return True

# HW 5
def partitionStats(instances, classIndex):
    total = len(instances)
    partitionDict = {}
    stats = []
    info = getCol(instances, classIndex)
    for val in info:
        partitionDict.setdefault(val,0)
        partitionDict[val] = partitionDict[val] + 1
    for key, val in partitionDict.items():
        stats.append([key,val,total, float(val)/total])
    return stats

# HW 5
def partitionInstances(instances, attIndex, attDomains):
    partitioned = {}
    for attDom in attDomains:
        partitioned.setdefault(attDom, [])
    for i in instances:
        partitioned[i[attIndex]].append(i)
    return partitioned

# HW 5        
def selectAttribute(instances, attIndexes, classIndex):
    Enews = [calcEnew(instances,attIndex,classIndex) for attIndex in attIndexes]
    return Enews.index(min(Enews))

# HW 5
def attFreqs(instances, attIndex, classIndex):
    attVals = list(set(getCol(instances, attIndex)))
    classVals = list(set(getCol(instances,classIndex)))
    result = {v:[{c:0 for c in classVals},0] for v in attVals}
    for row in instances:
        label = row[classIndex]
        attVal = row[attIndex]
        result[attVal][0][label] +=1
        result[attVal][1] += 1
    return result

# HW 5
def calcEnew(instances, attIndex, classIndex):
    results = attFreqs(instances, attIndex, classIndex)
    Enew = 0
    D=len(instances)
    for key,result in results.items():
        Dj = float(result[1])
        att = result[0]
        EDj = 0
        for K, attVal in att.items():
            value = (float(attVal) / Dj)
            EDj -= (value * math.log((value if value > 0 else 1),2))
        Enew += (float(Dj) / float(D)) * EDj
    return Enew

# HW 5
def tdidtClassifier(tree, instance, classIdx, isPrinting):
    predictedClass = getClassification(tree, instance)
    
    prediction = 0
    if predictedClass == "yes":
        prediction = 1
        
    actual = 0
    actRank = instance[classIdx]
    if actRank == "yes":
        actual = 1
    if isPrinting:
        print("class: "+mostProbRank+", actual: "+actRank)
    return (prediction,actual)

# HW 5 
def tdidtClassifierMPG(tree, instance, classIdx, isPrinting):
    prediction = int(getClassification(tree, instance))
    
    actual = int(instance[classIdx])
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

# HW 5
def getClassification(tree, instance):   
    idx = tree[0]
    
    if idx == 'Leaves':
        largestClass = ""
        largestProb = 0
        for i in range(1,len(tree)):
            if tree[i][3] > largestProb:
                largestProb = tree[i][3]
                largestClass = tree[i][0]
        return largestClass
    else:
        for i in range(1,len(tree)):
            if tree[i][0] == instance[idx]:
                return getClassification(tree[i][1], instance)

# HW 5
def setRankings(table, weightIdx, MPGIdx):
    newTable = []
    
    for row in table:
        rowTemp = row
        rowTemp[weightIdx] = str(getNHTSARanking(float(row[weightIdx])))
        rowTemp[MPGIdx] = str(getDOERanking(float(row[MPGIdx])))
        newTable.append(rowTemp)
    return newTable
     
###### END FUNCTIONS FROM PREVIOUS HOMEWORKS ######

# Get randomly stratified test and remainder sets
def randomStratify(table, class_idx):
    test = []
    remainder = []
    class_tables = {}
    
    for row in table:
        if row[class_idx] not in class_tables.keys():
            class_tables[row[class_idx]] = []
        class_tables[row[class_idx]].append(row)
        
    for k in class_tables.keys():
        shuffle(class_tables[k])
        
    for k in class_tables.keys():
        oneThird = int(len(class_tables[k]) / 3)
        test = test + class_tables[k][0:oneThird+1]
        remainder = remainder + class_tables[k][oneThird:]
    return (test, remainder)

# typeVar = whether titanic, auto, or wisconson
#           titanic = 0, auto = 1, wisconson = 2        
def createRandomForests(table, class_index, N, F, numClass, typeVar, atts, domains):
    forests = []
    for x in range(0,N):
        test, training = bootstrap(table)
        tree = randomTdidt(training, atts, domains, class_index, F)
        accuracy = determineAccuracy(tree,test, class_index, numClass, typeVar)
        forests.append([tree,accuracy])
    return forests
    
def bootstrap(table):
    test = []
    train = []
    idxs = [random.randint(0, len(table)-1) for _ in table]
    for i in range(0, len(table)):
        if i in idxs:
            train.append(table[i])
        else:
            test.append(table[i])
    return (test, train)
    
def randomTdidt(instances, atts, domains, classIndex, F):
    if len(domains) == 0:
        leaves = ['Leaves']
        leaves += partitionStats(instances, classIndex)
        return leaves
    elif len(domains) == 1:
        if sameClass(instances, classIndex):
            total = len(table)
            return ['Leaves',[instances[0][classIndex],total,total,1.0]]
        else:
            partitions = partitionInstances(instances, atts[0], domains[0])
            
            subtree = [atts[0]]
            for key, value in partitions.items():
                valueList = [key]
                if len(value) < 1:
                    valueList.append(tdidt(instances, [], [], classIndex))
                else:
                    valueList.append(tdidt(value, [], [], classIndex))
                subtree.append(valueList)
            return subtree
    else:
        partitionAttIdx = selectAttributeRandom(instances, atts, classIndex, F)
        partitions = partitionInstances(instances, atts[partitionAttIdx], domains[partitionAttIdx])
        
        subtree = [atts[partitionAttIdx]]
        for key, value in partitions.items():
            valueList = [key]
            if len(value) < 1:
                valueList.append(tdidt(instances, [], [], classIndex))
            else:
                valueList.append(tdidt(value, 
                    [att for att in atts if att != atts[partitionAttIdx]], 
                    [dom for dom in domains if dom != domains[partitionAttIdx]], classIndex))
            subtree.append(valueList)
        return subtree
        
def selectAttributeRandom(instances, attIndexes, classIndex, F):
    attIndexTemp = attIndexes[:]
    randIndexes = []
    if F >= len(attIndexes):
        randIndexes = attIndexes[:]
    else:
        shuffle(attIndexTemp)
        randIndexes = attIndexTemp[:F]
    Enews = [[calcEnew(instances,randIndex,classIndex),randIndex] for randIndex in randIndexes]
    sortedEnews = sorted(Enews, key=lambda x: x[0])
    return attIndexes.index(sortedEnews[0][1])

# typeVar = whether titanic, auto, or wisconson
#           titanic = 0, auto = 1, wisconson = 2    
def determineAccuracy(tree,test, class_index, numClass, typeVar):
    confusionMat = [[0 for p in range(0,numClass+2)] for l in range(0,numClass)]
    
    for t in test:
        if typeVar == 0:
            prediction, actual = tdidtClassifier(tree, t, class_index, 0)
            confusionMat[actual][prediction] += 1
        elif typeVar == 1:
            prediction, actual = tdidtClassifierMPG(tree, t, class_index, 0)
            confusionMat[actual-1][prediction-1] += 1
        else:
            prediction, actual = tdidtClassifierWisconsin(tree, t, class_index, 0)
            confusionMat[actual][prediction] += 1
            
    accuracy = calcAccuracy(confusionMat, numClass)
    
    return accuracy
    
# typeVar = whether titanic, auto, or wisconson
#           titanic = 0, auto = 1, wisconson = 2
def confusionMatrix(class_idx, numClass, test, forest, typeVar):
    confusionMat = [[0 for p in range(0,numClass+2)] for l in range(0,numClass)]
    
    for t in test:
        if typeVar == 0:
            prediction, actual = randomForestClassifierTitanic(forest, t, class_idx, 0)
        elif typeVar == 1:
            prediction, actual = randomForestClassifierMPG(forest, t, class_idx, 0)
        else:
            prediction, actual = randomForestClassifierWisconsin(forest, t, class_idx, 0)
        confusionMat[actual][prediction] += 1
            
    accuracy = calcAccuracy(confusionMat, numClass)
    
    for w in range(0, numClass):
        trues = confusionMat[w][w]
        total = sum(confusionMat[w])
        confusionMat[w][numClass] = trues
        if total > 0:
            confusionMat[w][numClass+1] = trues / (total*1.0) * 100
        else:
            confusionMat[w][numClass+1] = 0
        trues = confusionMat[w][w]
        total = sum(confusionMat[w])
        confusionMat[w][numClass] = trues
        if total > 0:
            confusionMat[w][numClass+1] = trues / (total*1.0) * 100
        else:
            confusionMat[w][+1] = 0
    
    outputMatrix = [[0 for p in range(0,numClass+3)] for l in range(0,numClass)]
    for i in range(0, numClass):
        for k in range(0, numClass+3):
            if k == 0:
                if typeVar == 1:
                    outputMatrix[i][k] = i+1
                elif typeVar == 0:
                    if i == 0:
                        outputMatrix[i][k] = "No"
                    else:
                        outputMatrix[i][k] = "Yes"
                else:
                    if i == 0:
                        outputMatrix[i][k] = "benign"
                    else:
                        outputMatrix[i][k] = "malignant"
            else:
                outputMatrix[i][k] = confusionMat[i][k-1]
    
    return (accuracy, outputMatrix)
    
def chooseMBest(forest, M):
    result = []
    for tree in forest:
        i = 0
        added = False
        if result == []:
            result.append(tree[0])
            added = True
        for r in result:
            if r[1] < tree[1]:
                result.insert(i,tree[0])
                added = True
                break
            i = i + 1
        if added == False:
            result.append(tree[0])
        if len(result) > M:
            result.pop()
    return result
    
def randomForestClassifierTitanic(forest, instance, classIdx, isPrinting):
    classDict = {}
    for tree in forest:
        predictedClass = getClassification(tree, instance)
        if predictedClass not in classDict.keys():
            classDict[predictedClass] = 0
        classDict[predictedClass] += 1
    
    mostProbClass = ""
    highestProb = 0
    
    for key in classDict.keys():
        if classDict[key] > highestProb:
            mostProbClass = key
            highestProb = classDict[key]
    
    prediction = 0
    if mostProbClass == "yes":
        prediction = 1
        
    actual = 0
    actRank = instance[classIdx]
    if actRank == "yes":
        actual = 1
    
    if isPrinting:
        print("class: "+mostProbClass+", actual: "+actRank)
    return (prediction,actual)
    
def randomForestClassifierMPG(forest, instance, classIdx, isPrinting):
    classDict = {}
    for tree in forest:
        predictedClass = getClassification(tree, instance)
        if predictedClass not in classDict.keys():
            classDict[predictedClass] = 0
        classDict[predictedClass] += 1
    
    mostProbClass = ""
    highestProb = 0
    
    for key in classDict.keys():
        if classDict[key] > highestProb:
            mostProbClass = key
            highestProb = classDict[key]
    
    prediction = mostProbClass
    actual = instance[classIdx]
    
    if isPrinting:
        print("class: "+prediction+", actual: "+actual)
    return (int(prediction)-1,int(actual)-1)

# Wisconsin
def randomForestClassifierWisconsin(forest, instance, classIdx, isPrinting):
    classDict = {}
    for tree in forest:
        predictedClass = getClassification(tree, instance)
        if predictedClass not in classDict.keys():
            classDict[predictedClass] = 0
        classDict[predictedClass] += 1
    
    mostProbClass = ""
    highestProb = 0
    
    for key in classDict.keys():
        if classDict[key] > highestProb:
            mostProbClass = key
            highestProb = classDict[key]

    predStr = "benign"
    prediction = 0
    if mostProbClass == '4':
        prediction = 1
        predStr = "malignant"
        
    actStr = "benign"
    actual = 0
    actualNum = instance[classIdx]
    if actualNum == '4':
        actual = 1
        actStr = "malignant"
    
    if isPrinting:
        print("class: "+predStr+", actual: "+actStr)
    return (int(prediction),int(actual))

# Wisconsin
def tdidtClassifierWisconsin(tree, instance, classIdx, isPrinting):
    predictedClass = getClassification(tree, instance)
    
    predStr = "benign"
    prediction = 0
    if predictedClass == '4':
        prediction = 1
        predStr = "malignant"
         
    actStr = "benign"
    actual = 0
    actualNum = instance[classIdx]
    if actualNum == '4':
         actual = 1
         actStr = "malignant"
    
    if isPrinting:
        print("class: "+predStr+", actual: "+actStr)
    return (int(prediction),int(actual))

def main():
    print("===========================================")
    print("STEP 2: Titanic Dataset")
    print("===========================================")
    
    # Load titanic table
    table = csvToTableSkipTitle("titanic.txt")
    
    #indices of attributes to predict based on
    # class = 0, age = 1, gender = 2
    atts = [0, 1, 2]
    domains = [["first","second","third","crew"],["adult","child"],["female","male"]]
    class_index = 3
    N = 20
    M = 7
    F = 2
    
    # To TEST N, M, F
    N = 60
    M = 7
    F = 2
    
    #RANDOM FORESTS
    test, remainder = randomStratify(table,class_index)
    forests = createRandomForests(table, class_index, N, F, 2, 0, atts, domains)
    finalForests = chooseMBest(forests, M)
    accuracy, outputMatrix = confusionMatrix(class_index, 2, test, finalForests, 0)

    print("Random Forest (On Titanic) : accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    print("Random Forest (On Titanic) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["Survival","No","Yes", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,4)] for l in range(0,2)]
    # Rules over entire dataset
    tree = tdidt(remainder, atts, domains, class_index)
    for testRow in test:
        # Descision Tree
        prediction, actual = tdidtClassifier(tree, testRow, class_index, 0)
        confusionMatrixDT[actual][prediction] += 1
    
    accuracy = calcAccuracy(confusionMatrixDT, 2)
    print("Normal Decision Tree (On Titanic) : accuracy: " + 
        str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    for w in range(0, 2):
        trues = confusionMatrixDT[w][w]
        total = sum(confusionMatrixDT[w])
        confusionMatrixDT[w][2] = trues
        if total > 0:
            confusionMatrixDT[w][3] = trues / (total*1.0) * 100
        else:
            confusionMatrixDT[w][3] = 0
        trues = confusionMatrixDT[w][w]
        total = sum(confusionMatrixDT[w])
        confusionMatrixDT[w][2] = trues
        if total > 0:
            confusionMatrixDT[w][3] = trues / (total*1.0) * 100
        else:
            confusionMatrixDT[w][3] = 0
    
    outputMatrix = [[0 for p in range(0,5)] for l in range(0,2)]
    for i in range(0, 2):
        for k in range(0, 5):
            if k == 0:
                if i == 0:
                    outputMatrix[i][k] = "No"
                else:
                    outputMatrix[i][k] = "Yes"
            else:
                outputMatrix[i][k] = confusionMatrixDT[i][k-1]
    print("Normal Decision Tree (On Titanic) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["Survival","No","Yes", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")

    
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
    tableOriginal = csvToTable("auto-data-meaningfulavg.txt")
    table = setRankings(tableOriginal, 4, 0)
    atts = [1, 4, 6]
    domains = [['8', '3', '5', '4', '6'],['1', '2', '3', '4', '5'],['77', '76', '75', '74', '73', '72', '71', '70', '79', '78']]
    class_index = 0
    
    print("===========================================")
    print("STEP 2: Auto Dataset")
    print("===========================================")
    
    #RANDOM FORESTS
    test, remainder = randomStratify(table,class_index)
    forests = createRandomForests(table, class_index, N, F, 10, 1, atts, domains)
    finalForests = chooseMBest(forests, M)
    accuracy, outputMatrix = confusionMatrix(class_index, 10, test, finalForests, 1)

    print("Random Forest (On Auto) : accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    print("Random Forest (On Auto) :")
    tableView = tabulate.tabulate(
        outputMatrix,headers=["MPG","1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,12)] for l in range(0,10)]
    # Rules over entire dataset
    tree = tdidt(remainder, atts, domains, class_index)

    for testRow in test:
        # Descision Tree
        prediction, actual = tdidtClassifierMPG(tree, testRow, class_index, 0)
        confusionMatrixDT[actual-1][prediction-1] += 1

    accuracy = calcAccuracy(confusionMatrixDT, 10)
    print("Normal Decision Tree (On Auto) : accuracy: " + 
        str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    for w in range(0, 10):
        trues = confusionMatrixDT[w][w]
        total = sum(confusionMatrixDT[w])
        confusionMatrixDT[w][10] = trues
        if total > 0:
            confusionMatrixDT[w][11] = trues / (total*1.0) * 100
        else:
            confusionMatrixDT[w][11] = 0
    
    outputMatrix = [[0 for p in range(0,13)] for l in range(0,10)]
    for i in range(0, 10):
        for k in range(0, 13):
            if k == 0:
                outputMatrix[i][k] = i+1
            else:
                outputMatrix[i][k] = confusionMatrixDT[i][k-1]
    print("Normal Decision Tree (On Auto) :")
    tableView = tabulate.tabulate(
        outputMatrix,headers=["MPG","1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    print("===========================================")
    print("STEP 3: Wisconsin Dataset")
    print("===========================================")
    
    # Load wisconsin table
    tableOriginal = csvToTable("wisconsin.dat")
    table = []
    for row in tableOriginal:
        newRow = []
        for a in row:
            newA = a.split('\r')
            newRow.append(newA[0])
        table.append(newRow)
    
    #indices of attributes to predict based on
    atts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    numList = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    domains = [numList, numList, numList, numList, numList, numList, numList, numList, numList]
    class_index = 9
    N = 60
    M = 30
    F = 2

    #RANDOM FORESTS
    test, remainder = randomStratify(table,class_index)
    forests = createRandomForests(table, class_index, N, F, 2, 2, atts, domains)
    finalForests = chooseMBest(forests, M)
    accuracy, outputMatrix = confusionMatrix(class_index, 2, test, finalForests, 2)

    print("Random Forest (On Wisconsin) : accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    print("Random Forest (On Wisconsin) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["Tumor","benign","malignant", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,12)] for l in range(0,10)]
    # Rules over entire dataset
    tree = tdidt(remainder, atts, domains, class_index)

    for testRow in test:
        # Descision Tree
        prediction, actual = tdidtClassifierWisconsin(tree, testRow, class_index, 0)
        confusionMatrixDT[actual][prediction] += 1

    accuracy = calcAccuracy(confusionMatrixDT, 2)
    print("Normal Decision Tree (On Wisconsin) : accuracy: " + 
        str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    for w in range(0, 2):
        trues = confusionMatrixDT[w][w]
        total = sum(confusionMatrixDT[w])
        confusionMatrixDT[w][2] = trues
        if total > 0:
            confusionMatrixDT[w][3] = trues / (total*1.0) * 100
        else:
            confusionMatrixDT[w][3] = 0
        trues = confusionMatrixDT[w][w]
        total = sum(confusionMatrixDT[w])
        confusionMatrixDT[w][2] = trues
        if total > 0:
            confusionMatrixDT[w][3] = trues / (total*1.0) * 100
        else:
            confusionMatrixDT[w][3] = 0
    
    outputMatrix = [[0 for p in range(0,5)] for l in range(0,2)]
    for i in range(0, 2):
        for k in range(0, 5):
            if k == 0:
                if i == 0:
                    outputMatrix[i][k] = "benign"
                else:
                    outputMatrix[i][k] = "malignant"
            else:
                outputMatrix[i][k] = confusionMatrixDT[i][k-1]
    print("Normal Decision Tree (On Wisconsin) :")
    tableView = tabulate.tabulate(
        outputMatrix,headers=["Tumor","benign","malignant", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")

if __name__ == '__main__':
    main()
