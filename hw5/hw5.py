#!/usr/bin/python
# CPSC 462 Data Mining
# HW5
# Kristina Spring, Myanna Harris
# Nov 2, 2016
# Decision Trees

import math
import random
import tabulate
import numpy
#import graphviz as gv

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
     
###### END FUNCTIONS FROM PREVIOUS HOMEWORKS ######

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

def sameClass(instances, classIndex):
    classValue = instances[0][classIndex]
    for instance in instances:
        if instance[classIndex] == classValue:
            return False
    return True

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

def partitionInstances(instances, attIndex, attDomains):
    partitioned = {}
    for attDom in attDomains:
        partitioned.setdefault(attDom, [])
    for i in instances:
        partitioned[i[attIndex]].append(i)
    return partitioned
        

def selectAttribute(instances, attIndexes, classIndex):
    Enews = [calcEnew(instances,attIndex,classIndex) for attIndex in attIndexes]
    return Enews.index(min(Enews))

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
    
def tdidtClassifierMPG(tree, instance, classIdx, isPrinting):
    prediction = int(getClassification(tree, instance))
    
    actual = int(instance[classIdx])
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

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
    
def printRules(tree):
    print("Rules: ")
    queue = []
    printTree(tree, queue)
    
def printTree(tree, queue):
    if len(tree) < 2:
        return ""
        
    idx = tree[0]
    
    if idx == 'Leaves':
        largestClass = ""
        largestProb = 0
        for i in range(1,len(tree)):
            if tree[i][3] > largestProb:
                largestProb = tree[i][3]
                largestClass = tree[i][0]
        print "IF att1 = " + str(queue[0]),
        for i in range(1, len(queue)):
            print "and att"+str(i+1)+" = " + str(queue[i]),
        
        print "THEN class = " + str(largestClass)
    else:
        for i in range(1,len(tree)):
            queue.append(tree[i][0])
            printTree(tree[i][1], queue)
            del queue[-1]

def setRankings(table, weightIdx, MPGIdx):
    newTable = []
    
    for row in table:
        rowTemp = row
        rowTemp[weightIdx] = str(getNHTSARanking(float(row[weightIdx])))
        rowTemp[MPGIdx] = str(getDOERanking(float(row[MPGIdx])))
        newTable.append(rowTemp)
    return newTable

def main():
    
    print("===========================================")
    print("STEP 1: Titanic Dataset")
    print("===========================================")
    
    # Load titanic table
    table = csvToTableSkipTitle("titanic.txt")
    
    #indices of attributes to predict based on
    # class = 0, age = 1, gender = 2
    atts = [0, 1, 2]
    domains = [["first","second","third","crew"],["adult","child"],["female","male"]]
    class_index = 3
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamplesGen(table, 10, 3)

    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,4)] for l in range(0,2)]

    for i in range(0, 10):
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]
        
        # Decision Tree
        # Rules over entire dataset
        tree = tdidt(tableTrain, atts, domains, class_index)

        for testRow in tableTest:
            # Descision Tree
            prediction, actual = tdidtClassifier(tree, testRow, class_index, 0)
            confusionMatrixDT[actual][prediction] += 1

    accuracy = calcAccuracy(confusionMatrixDT, 2)
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

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
                
    print("Descision Tree (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["Survival","No","Yes", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    # Rules over entire dataset
    tree = tdidt(table, atts, domains, class_index)
    printRules(tree)

    
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
    print("STEP 2: Car Dataset")
    print("===========================================")
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamples(table, 10)

    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,12)] for l in range(0,10)]

    for i in range(0, 10):
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]
        
        # Decision Tree
        # Rules over entire dataset
        tree = tdidt(tableTrain, atts, domains, class_index)


        for testRow in tableTest:
            # Descision Tree
            prediction, actual = tdidtClassifierMPG(tree, testRow, class_index, 0)
            confusionMatrixDT[actual-1][prediction-1] += 1

    accuracy = calcAccuracy(confusionMatrixDT, 10)
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))

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
    
    print("Descision Tree (Stratified 10-Fold cross Validation Results) :")
    tableView = tabulate.tabulate(outputMatrix,headers=["MPG","1","2","3","4","5","6", "7", "8", "9", "10", "Total", "Recognition (%)"],tablefmt="rst")
    print(tableView)
    print("")
    
    # Rules over entire dataset
    tree = tdidt(table, atts, domains, class_index)
    printRules(tree)


if __name__ == '__main__':
    main()
