#!/usr/bin/python

import tabulate
 

def findDuplicates(rows, key):
    prevRows = set()
    duplicates = []
    for row in rows:
        splitRow = row.split(",")
        checkString = ",".join([splitRow[k] for k in key])
        if checkString in prevRows:
            duplicates.append(checkString)
        else:
            prevRows.add(checkString)
    return duplicates

def numInstances(rows):
    return len(rows)

def printInfo(title, fileName, key):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\n"))
    print("--------------------")
    print(title+": "+fileName)
    print("--------------------")
    print("No. of instances: "+str(numInstances(rows)))
    print("Duplicates: "+str(findDuplicates(rows, key)))
    print("")
    file.close()

def convertToDict(fileName, key):
    dictionary = {}
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\n"))
    for row in rows:
        splitRow = row.split(",")
        dKey = ",".join([splitRow[k] for k in key])
        value = splitRow
        dictionary[dKey] = value
    file.close()
    return dictionary

def fullOuterJoin(d1, k1, d2, k2):
    outList = []
    keysList = d1.keys()
    for k in keysList:
        if d2.has_key(k):
            d2Val = ""
            for i in range(0,len(d2[k])):
                if i not in k2:
                    d2Val = "," + d2[k][i]
            outList.append(",".join(d1[k])+d2Val)
            del d2[k]
        else:
            outList.append(",".join(d1[k])+",NA")
    for k in d2.keys():
        d2Val = ""
        for i in range(0,len(d2[k])):
            if i not in k2:
                d2Val = "," + d2[k][i]
        outList.append("NA,NA,NA,NA,NA,NA,"+d2[k][k2[0]]+",NA,"+d2[k][k2[1]]+d2Val)
    return outList

def sendToFile(fileName, joinedList):
    file = open(fileName, "w")
    for r in joinedList:
        file.write(r+"\n")
    file.close()

def summaryStats(fileName, attr):
    file = open(fileName, "r")
    rows = [r.split(",") for r in filter(None, file.read().split("\n"))]
    file.close()
    results = []
    for key in attr.keys():
        val = attr[key]
        numList = sorted([float(r[key]) for r in rows if (r[key]!="NA")])
        rRow = [val, getMin(numList), getMax(numList), getMid(numList), getAvg(numList), getMed(numList)]
        results.append(rRow)
    tableView = tabulate.tabulate(results,headers=["attribute","min","max","mid","avg","med"],tablefmt="rst")
    print(tableView)
    return results

def getMin(list):
    return list[0]

def getMax(list):
    return list[len(list)-1]

def getMid(list):
    return (list[0]+list[len(list)-1])/2.0

def getAvg(list):
    total = 0
    for x in list:
        total += x
    return total/len(list)

def getMed(list):
    length = len(list)
    if length % 2 == 0:
        return (list[length/2] + list[length/2-1])/2.0
    else:
        return list[length/2]

def removeIncompleteRows(origFile, newFile):
    file = open(origFile, "r")
    rows = [r.split(",") for r in filter(None, file.read().split("\n"))]
    file.close()
    newInfo = [",".join(row) for row in rows if ("NA" not in row)]
    file = open(newFile, "w")
    file.write("\n".join(newInfo))
    file.close()

def replaceMissingWAvg(origFile, newFile, attr, stats):
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
                r[key] = str(stats[i][4])
    for r in rows:
        out.append(",".join(r))
    file = open(newFile, "w")
    file.write("\n".join(out))
    file.close()

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

def main():
    printInfo("", "auto-mpg-nodups.txt", [8,6])
    printInfo("", "auto-prices-nodups.txt", [0,1])
    autoMPG = convertToDict("auto-mpg-nodups.txt", [8,6])
    autoPrices = convertToDict("auto-prices-nodups.txt", [0,1])
    joined = fullOuterJoin(autoMPG, [8,6], autoPrices,[0,1])
    sendToFile("auto-data.txt", joined)
    printInfo("combined table (saved as auto-data.txt)", "auto-data.txt", [8,6])
    contDict = {0:"MPG",2:"displacement",3:"horsepower",4:"weight",5:"acceleration",9:"MSRP"}
    origResults = summaryStats("auto-data.txt", contDict)
    removeIncompleteRows("auto-data.txt", "auto-data-missingvalsremoved.txt")
    printInfo("combined table (rows w/ missing values removed)", "auto-data-missingvalsremoved.txt", [8,6])
    summaryStats("auto-data-missingvalsremoved.txt", contDict)
    replaceMissingWAvg("auto-data.txt", "auto-data-replacemissing.txt", contDict, origResults)
    printInfo("combined table (rows w/ missing values replaced with average)", "auto-data-replacemissing.txt", [8,6])
    summaryStats("auto-data-replacemissing.txt", contDict)
    replaceMissingWMeaningfulAvg("auto-data.txt", "auto-data-meaningfulavg.txt", contDict)
    printInfo("combined table (rows w/ missing values replaced with meaningful average)", "auto-data-meaningfulavg.txt", [8,6])
    summaryStats("auto-data-meaningfulavg.txt", contDict)

   
if __name__ == "__main__":
    main()

