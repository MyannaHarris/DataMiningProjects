#!/usr/bin/python
# CPSC 462 Data Mining
# HW7
# Kristina Spring, Myanna Harris
# Dec 3, 2016
# Apriori

import math
import random
import tabulate
import numpy
from random import shuffle
import itertools

###### FUNCTIONS FROM PREVIOUS HOMEWORKS ######

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

###### END FUNCTIONS FROM PREVIOUS HOMEWORKS ######

# gets rules
# Rule = [L, R]
# Rule = [[[idx1, val1], ...], [[idx2, val2], ...]]
# Interestingness measure = [support, confidence, lift]
def getAssociationRules(minConf, minSupp, dataset, domains, attIdxs):
    rules = []
    interestMeasures = []
    
    # get rules
    aprioriSets = getAprioriSets(dataset, domains, attIdxs, minSupp)
    rules = getRulesFromSets(aprioriSets, dataset, minConf)
    
    # get interestingness measures
    for rule in rules:
        measures = []
        measures.append(getSupport(dataset, rule))
        measures.append(getConfidence(dataset, rule))
        measures.append(getLift(dataset, rule))
        interestMeasures.append(measures)
    
    # get rid of rules that aren't good enough
    i = 0
    while i < len(rules):
        measures = interestMeasures[i]
        if measures[0] < minSupp or measures[1] < minConf:
            del rules[i]
            del interestMeasures[i]
            i -= 1
        i += 1
    
    return (rules, interestMeasures)

# gets rules
# Rule = [L, R]
# Rule = [[(idx1, val1), ...], [(idx2, val2), ...]]
def getRulesFromSets(aprioriSets, dataset, minConf):
    rules = []
    badSets = []
    
    for i in range(1, len(aprioriSets)):
        L = aprioriSets[i]
        for setL in L:
            if len(setL) < 3:
                rule = []
                rule.append([setL[0]])
                rule.append([setL[1]])
                if getConfidence(dataset, rule) >= minConf:
                    rules.append(rule)
                    
                rule = []
                rule.append([setL[1]])
                rule.append([setL[0]])
                if getConfidence(dataset, rule) >= minConf:
                    rules.append(rule)
                        
            for r in range(1, len(setL)-1):
                perm_iterator = itertools.combinations(setL, r)
                doneWithSet = True

                for item in perm_iterator:
                    doneWithItem = False
                    itemLst = list(item)
                    for a in badSets:
                        if a.issubset(itemLst):
                            doneWithItem = True
                    if not doneWithItem:
                        rule = []
                        rule.append([lhs for lhs in setL if lhs not in item])
                        rule.append(itemLst)
                        if getConfidence(dataset, rule) >= minConf:
                            rules.append(rule)
                            doneWithSet = False
                        else:
                            s = set()
                            for sItem in itemLst:
                                s.add(sItem)
                            badSets.append(s)
                        
                if doneWithSet:
                    break
    
    return rules

# gets sets
# sets = [[[(attribute), itemset], L(i) set of itemsets size i], all Ls]
# sets = [[[(idx1, val1), (idx2, val2), ...], ...], ...]
def getAprioriSets(dataset, domains, attIdxs, minSupp):
    sets = []
    # L1 at index 0
    L1 = []
    c1 = []
    for i in range(0, len(attIdxs)):
        for d in domains[attIdxs[i]]:
            item = (attIdxs[i], d)
            c1.append([item])
         
    minNumOccurs = len(dataset) * minSupp
                 
    for c in c1:
        item = c[0]
        if occursEnough(dataset, item[0], item[1], minNumOccurs):
            L1.append(c)
    sets.append(L1)
    
    k = 0
    while not(sets[k] == []):
        oldL = sets[k]
        newL = []
        
        for i in range(0, len(oldL) - 1):
            for j in range(i + 1, len(oldL)):
                itemset1 = oldL[i]
                itemset2 = oldL[j]
                
                if ((itemset1[:-1] == itemset2[:-1]) and 
                    not (itemset1[-1][0] == itemset2[-1][0])):
                    newItemset = itemset1[:-1]
                    if itemset1[-1][0] < itemset2[-1][0]:
                        newItemset.append(itemset1[-1])
                        newItemset.append(itemset2[-1])
                    else:
                        newItemset.append(itemset2[-1])
                        newItemset.append(itemset1[-1])
                        
                    if checkItemset(k_1_subsets(newItemset), oldL):
                        if setOccursEnough(dataset, newItemset, minNumOccurs):
                            newL.append(newItemset)
        
        sets.append(newL)
        k += 1
    
    return sets

# check if itemset occurs enough
def setOccursEnough(dataset, itemset, minNum):
    numOccur = 0
    for r in dataset:
        for i in itemset:
            if r[i[0]] == i[1]:
                numOccur += 1
                if minNum <= numOccur:
                    return True
                
    return False

# check if itemset's subsets are all in L(k-1)
def checkItemset(subset, L):
    for s in subset:
        if s not in L:
            return False
            
    return True

# get subsets of size k-1
def k_1_subsets(itemset):
    return [itemset[:i] + itemset[i+1:] for i in range(0, len(itemset))]

# get whether value occurs enough
def occursEnough(dataset, idx, val, minNum):
    numOccur = 0
    for r in dataset:
        if r[idx] == val:
            numOccur += 1
            if minNum <= numOccur:
                return True
                
    return False

# get support    
def getSupport(dataset, rule):
    support = 0
    allRules = rule[0] + rule[1]
    
    for row in dataset:
        supported = True
        for r in allRules:
            if not (row[r[0]] == r[1]):
                supported = False
                break
        if supported:
            support += 1
    
    if len(dataset) == 0:
        return 0.0
    else:
        return float(support) / len(dataset)

# get support of one side of rule   
def getSupportOneSide(dataset, ruleSide):
    support = 0
    
    for row in dataset:
        supported = True
        for r in ruleSide:
            if not (row[r[0]] == r[1]):
                supported = False
                break
        if supported:
            support += 1
    
    if len(dataset) == 0:
        return 0.0
    else:
        return float(support) / len(dataset)

# get confidence  
def getConfidence(dataset, rule):
    confidence = 0
    countS = 0
    countL = 0
    
    for row in dataset:
        supported = True
        leftSupported = True
        
        for r in rule[0]:
            if not (row[r[0]] == r[1]):
                supported = False
                leftSupported = False
                break
        
        for r in rule[1]:
            if not (row[r[0]] == r[1]):
                supported = False
                break
                
        if supported:
            countS += 1
        if leftSupported:
            countL += 1
            
    if countL == 0:
        confidence = 0.0
    else:
        confidence = float(countS) / countL
    
    return confidence

# get lift
def getLift(dataset, rule):
    lift = 0
    countS = getSupport(dataset, rule)
    countL = getSupportOneSide(dataset, rule[0])
    countR = getSupportOneSide(dataset, rule[1])
    
    if (countL * countR) == 0:
        lift = 0.0
    else:
        lift = float(countS) / (countL * countR)
    
    return lift
    
def displayRules(rules, measures, attNames, attVals):
    outputs = []

    for i in range(0, len(rules)):
        output = []
        rule = rules[i]
        
        ruleOut = str(i) + " "
        
        # left side
        for k in range(0, len(rule[0])):
            r = rule[0][k]
            if not (attVals == []):
                ruleOut += attNames[r[0]] + "=" + attVals[r[0]][r[1]]
            else:
                ruleOut += attNames[r[0]] + "=" + r[1]
            if k < len(rule[0]) - 1:
                ruleOut += " and "
                
        ruleOut += " => "
        
        # right side
        for k in range(0, len(rule[1])):
            r = rule[1][k]
            if not (attVals == []):
                ruleOut += attNames[r[0]] + "=" + attVals[r[0]][r[1]]
            else:
                ruleOut += attNames[r[0]] + "=" + r[1]
            if k < len(rule[1]) - 1:
                ruleOut += " and "
                
        output.append(ruleOut)
        
        measure = measures[i]
        for m in measure:
            output.append(m)
        
        outputs.append(output)
            
    tableView = tabulate.tabulate(outputs, 
        headers=["association rule", "support", "confidence", "lift"])
    print(tableView)

def main():
    print("===========================================")
    print("Titanic Dataset")
    print("===========================================")
    
    # Load titanic table
    table = csvToTableSkipTitle("titanic.txt")
    
    # attribute info
    # class = 0, age = 1, gender = 2
    atts = [0, 1, 2]
    domains = [["first","second","third","crew"],["adult","child"],["female","male"]]
    class_index = 3
    all_atts = [0, 1, 2, 3]
    all_domains = [["first","second","third","crew"],["adult","child"],
        ["female","male"], ["yes", "no"]]
    attNames = ["Class", "Age", "Gender", "Survival"]
    minConf = 0.9
    minSupp = 0.9
    
    rules, measures = getAssociationRules(minConf, minSupp, table,
        all_domains, all_atts)
    displayRules(rules, measures, attNames, [])

    print("===========================================")
    print("Mushroom Dataset")
    print("===========================================")
    
    # load mushroom dataset
    tableMushroom = csvToTable("agaricus-lepiota.txt")
    
    # attribute info
    class_index = 0
    all_atts = [i for i in range(0, 23)]
    all_domains = [['e', 'p'], ['b','c','x','f','k','s'],['f','g','y','s'],
        ['n','b','c','g','r','p','u','e','w','y'],['t','f'],
        ['a','l','c','y','f','m','n','p','s'],['a','d','f','n'],['c','w','d'],
        ['b','n'],['k','n','b','h','g','r','o','p','u','e','w','y'],['e','t'],
        ['b','c','u','e','z','r'],['f','y','k','s'],['f','y','k','s'],
        ['n','b','c','g','o','p','e','w','y'],
        ['n','b','c','g','o','p','e','w','y'],['p','u'],['n','o','w','y'],
        ['n','o','t'],['c','e','f','l','n','p','s','z'],
        ['k','n','b','h','r','o','u','w','y'],['a','c','n','s','v','y'],
        ['g','l','m','p','u','w','d']]
    attNames = ["Edibleness", 
        "Cap-shape",
        "Cap-surface",
        "Cap-color",
        "Bruises",
        "Odor",
        "Gill-attachment",
        "Gill-spacing",
        "Gill-size",
        "Gill-color",
        "Stalk-shape",
        "Stalk-root",
        "Stalk-surface-above-ring",
        "Stalk-surface-below-ring",
        "Stalk-color-above-ring",
        "Stalk-color-below-ring",
        "Veil-type",
        "Veil-color",
        "Ring-number",
        "Ring-type",
        "Spore-print-color",
        "Population",
        "Habitat"]
    atts = [0, 5, 6, 8, 10, 11]
        
    attVals = []
    
    attVals= [
        {'e' : "edible", 'p' : "poisonous"},
        {'b':"bell", 'c':"conical", 'x':"convex",  'f':"at", 'k':"knobbed", 
        's':"sunken"},
        {'f':"brous", 'g':"grooves", 'y':"scaly", 's':"smooth"},
        {'n':"brown", 'b':"bu", 'c':"cinnamon", 'g':"gray", 'r':"green", 
        'p':"pink", 'u':"purple", 'e':"red", 'w':"white", 'y':"yellow"},
        {'t':"bruises", 'f':"no"},
        {'a':"almond", 'l':"anise", 'c':"creosote",  'y':"shy", 'f':"foul", 
        'm':"musty", 'n':"none", 'p':"pungent", 's':"spicy"},
        {'a':"attached", 'd':"descending", 'f':"free", 'n':"notched"},
        {'c':"close", 'w':"crowded", 'd':"distant"},
        {'b':"broad", 'n':"narrow"},
        {'k':"black", 'n':"brown", 'b':"bu", 'h':"chocolate", 'g':"gray", 
        'r':"green", 'o':"orange", 'p':"pink", 'u':"purple", 'e':"red", 
        'w':"white", 'y':"yellow"},
        {'e':"enlarging", 't':"tapering"},
        {'b':"bulbous", 'c':"club", 'u':"cup", 'e':"equal", 'z':"rhizomorphs", 
        'r':"rooted"},
        {'f':"brous", 'y':"scaly", 'k':"silky", 's':"smooth"},
        {'f':"brous", 'y':"scaly", 'k':"silky", 's':"smooth"},
        {'n':"brown", 'b':"bu", 'c':"cinnamon", 'g':"gray", 'o':"orange", 
        'p':"pink", 'e':"red", 'w':"white", 'y':"yellow"},
        {'n':"brown", 'b':"bu", 'c':"cinnamon", 'g':"gray", 'o':"orange", 
        'p':"pink", 'e':"red", 'w':"white", 'y':"yellow"},
        {'p':"partial", 'u':"universal"},
        {'n':"brown", 'o':"orange", 'w':"white", 'y':"yellow"},
        {'n':"none", 'o':"one", 't':"two"},
        {'c':"cobwebby", 'e':"evanescent",  'f':"aring", 'l':"large", 
        'n':"none", 'p':"pendant", 's':"sheathing", 'z':"zone"},
        {'k':"black", 'n':"brown", 'b':"bu", 'h':"chocolate", 'r':"green", 
        'o':"orange", 'u':"purple", 'w':"white", 'y':"yellow"},
        {'a':"abundant", 'c':"clustered", 'n':"numerous", 's':"scattered", 
        'v':"several", 'y':"solitary"},
        {'g':"grasses", 'l':"leaves", 'm':"meadows", 'p':"paths", 'u':"urban", 
        'w':"waste", 'd':"woods"}]
    
    minConf = 0.7
    minSupp = 0.1
    
    rules, measures = getAssociationRules(minConf, minSupp, 
        tableMushroom, all_domains, atts)
    displayRules(rules, measures, attNames, attVals)
    

if __name__ == '__main__':
    main()
