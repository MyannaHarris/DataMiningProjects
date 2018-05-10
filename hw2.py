#!/usr/bin/python
# CPSC 462 Data Mining
# HW2
# Kristina Spring, Myanna Harris
# Sep 24, 2016

import matplotlib.pyplot as pyplot
import numpy
import math

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

def csvToTable(fileName):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\n"))
    file.close()
    table = []
    for row in rows:
        splitRow = row.split(",")
        table.append(splitRow)
    return table

def getCol(table, index):
    return [ row[index] for row in table ]

# labelFlag
# 0 = normal
# 1 = bins with last bin everything greater than last border
# 2 = bins with all bin borders listed
def makeBarChart(lstIn, filename, title, xlabel, labelFlag, labels):
    lst = sorted([float(x) for x in lstIn])
    x = []
    y = []
    for v in lst:
        if len(x) < 1 or x[len(x)-1] != v:
            x.append(v)
            y.append(1)
        else: 
            y[len(y)-1] += 1
    xrng = numpy.arange(len(x))
    pyplot.bar(xrng, y, 0.45, align='center')
    if labelFlag == 0:
        pyplot.xticks(xrng, x)
    else:
        labelLst = []
        labelLst.append("<="+str(labels[0]))
        for i in range(1,len(labels)):
            labelLst.append(str(labels[i-1]) + "-"+str(labels[i]))
        if labelFlag == 1:
            labelLst.append(">="+str(labels[len(labels)-1]+1))
        pyplot.xticks(xrng, labelLst)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel('Count')
    if max(y)%10 == 0:
        yrng = numpy.arange(0, max(y)+11, 10)
        pyplot.yticks(yrng, yrng)
    pyplot.grid(True)
    pyplot.savefig(filename)
    pyplot.figure()

def makePieChart(lstIn, filename, title):
    lst = sorted([float(x) for x in lstIn])
    x = []
    y = []
    for v in lst:
        if len(x) < 1 or x[len(x)-1] != v:
            x.append(v)
            y.append(1)
        else: 
            y[len(y)-1] += 1
    colors = ['r', 'chartreuse', 'yellow', 'c', 'lime', 'purple', 'orangered', 'm', 'orange', 'green']
    xrng = numpy.arange(len(x))
    pyplot.pie(y, labels=x, autopct='%1.1f%%', colors=colors)
    pyplot.title(title)
    pyplot.savefig(filename)
    pyplot.figure()

def makeDotChart(lstIn, filename, title, xlabel):
    lst = sorted([float(x) for x in lstIn])
    x = lst
    y = [1]*len(x)
    pyplot.plot(x, y, 'b.', alpha=0.2, markersize=16)
    pyplot.gca().get_yaxis().set_visible(False)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.savefig(filename)
    pyplot.figure()

def getMpgRatings(lst):
    results = []
    for val in lst:
        if float(val) <= 13:
            results.append(1)
        elif float(val) == 14:
            results.append(2)
        elif float(val) <= 16:
            results.append(3)
        elif float(val) <= 19:
            results.append(4)
        elif float(val) <= 23:
            results.append(5)
        elif float(val) <= 26:
            results.append(6)
        elif float(val) <= 30:
            results.append(7)
        elif float(val) <= 36:
            results.append(8)
        elif float(val) <= 44:
            results.append(9)
        elif float(val) > 44:
            results.append(10)
    return results

def getRightBordersList(lstIn, bins):
    borders = []
    lst = sorted([float(x) for x in lstIn])
    maxVal = lst[len(lst)-1]
    size = (maxVal - lst[0])/(bins*1.0)
    border = lst[0]+size
    while border <= maxVal:
        borders.append(border)
        border += size
    if borders[len(borders)-1] < maxVal:
        borders[len(borders)-1] = maxVal
    return borders

def getMPGRatingsWithBorders(lstIn, borders):
    lst = sorted([float(x) for x in lstIn])
    borderIdx = 0
    results = []
    for val in lst:
        if val <= borders[borderIdx]:
            results.append(borderIdx)
        else:
            while val > borders[borderIdx]:
                borderIdx += 1
            results.append(borderIdx)
    return results

def makeHistogram(lstIn, filename, title, xlabel):
    lst = sorted([float(x) for x in lstIn])
    x = lst

    pyplot.hist(x, bins=10, alpha=0.75, color = 'b')

    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel('Count')
    pyplot.grid(True)
    pyplot.savefig(filename)
    pyplot.figure()

def makeScatterPlot(table, xIdx, yIdx, filename, title, xlabel, ylabel, regVals):
    x = []
    y = []
    for row in table:
        x.append(float(row[xIdx]))
        y.append(float(row[yIdx]))

    maxX = max(x)
    maxY = max(y)
    pyplot.xlim(0, int(maxX * 1.10))
    pyplot.ylim(0, int(maxY * 1.10))
    pyplot.plot(x,y,'b.')
    if regVals != None:
        pyplot.plot(x, regVals[0]*numpy.asarray(x)+regVals[1])
        output = "Corr: " + str(regVals[2])+ " Cov: " + str(regVals[3])
        b = dict(facecolor='none', color='r')
        pyplot.text(maxX/40, maxY-(maxY/20), output, bbox=b, fontsize=10, color='r')
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.grid(True)
    pyplot.savefig(filename)
    pyplot.figure()

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

def makeBoxPlot(table):
    title = "MPG By Model Year"
    labels = [int(r[6]) for r in table]
    labels = list(set(labels))
    vals = [ [] for i in range(len(labels)) ]
    for row in table:
        index = labels.index(int(row[6]))
        vals[index].append(float(row[0]))
    pyplot.boxplot(vals)
    pyplot.title(title)
    pyplot.xlabel("Model Year")
    pyplot.ylabel("Miles Per Gallon")
    pyplot.xticks(numpy.arange(1,len(labels)+1,1),labels)
    pyplot.grid(True)
    pyplot.savefig("step-8-boxplot.pdf")
    pyplot.figure()

def makeMultFreq(table):
    title = "Cars From Each Country of Origin by Model Year"
    labels = [int(r[6]) for r in table]
    labels = list(set(labels))

    tables = [[] for k in range(3)]
    for r in table:
        if r[7] == "1":
            tables[0].append(r)
        elif r[7] == "2":
            tables[1].append(r)
        else:
            tables[2].append(r)

    # create a figure and one subplot
    # returns figure and x-axis
    fig, ax = pyplot.subplots()
    # create two bars (returns rectangle objects)
    xs = numpy.arange(1,len(labels)+1,1)
    vals = [ 0 for i in range(len(labels)) ]
    for row in tables[0]:
        index = labels.index(int(row[6]))
        vals[index] += 1
    r1 = ax.bar([n-0.3 for n in xs], vals, 0.2, color='black')
    vals = [ 0 for i in range(len(labels)) ]
    for row in tables[1]:
        index = labels.index(int(row[6]))
        vals[index] += 1
    r2 = ax.bar([n-0.1 for n in xs], vals, 0.2, color='grey')
    vals = [ 0 for i in range(len(labels)) ]
    for row in tables[2]:
        index = labels.index(int(row[6]))
        vals[index] += 1
    r3 = ax.bar([n+0.1 for n in xs], vals, 0.2, color='lightgrey')
    # create a legend, location upper left
    ax.legend((r1[0], r2[0], r3[0]), ('US', 'Europe', 'Japan'), loc=1)
    # set x value labels
    pyplot.xticks(numpy.arange(1,len(labels)+1,1),labels)

    pyplot.title(title)
    pyplot.xlabel("Model Year")
    pyplot.ylabel("Count")
    pyplot.grid(True)
    pyplot.savefig("step-8-multiFrequency.pdf")

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

    # Make bar charts for categorical attributes
    makeBarChart(getCol(table,1), "step-1-cylinders.pdf", "Cylinder Frequency", "Cylinders", 0, [])
    makeBarChart(getCol(table,6), "step-1-modelyear.pdf", "Model Year Frequency", "Model Year", 0, [])
    makeBarChart(getCol(table,7), "step-1-origin.pdf", "Origin Frequency", "Origin", 0, [])

    # Make pie charts for categorical attributes
    makePieChart(getCol(table,1), "step-2-cylinders.pdf", "Total Number of Cars by Number of Cylinders")
    makePieChart(getCol(table,6), "step-2-modelyear.pdf", "Total Number of Cars by Model Year")
    makePieChart(getCol(table,7), "step-2-origin.pdf", "Total Number of Cars by Origin")

    # Make dot charts for continuous attributes
    makeDotChart(getCol(table,0), "step-3-mpg.pdf", "Miles per Gallon of All Cars", "Miles per gallon")
    makeDotChart(getCol(table,2), "step-3-displacement.pdf", "Displacement of All Cars", "Displacement (cubic inches)")
    makeDotChart(getCol(table,3), "step-3-horsepower.pdf", "Horsepower of All Cars", "Horsepower (1 = 550 foot lbs per second)")
    makeDotChart(getCol(table,4), "step-3-weight.pdf", "Weight of All Cars", "Weight (lbs)")
    makeDotChart(getCol(table,5), "step-3-acceleration.pdf", "Acceleration of All Cars", "Acceleration (seconds to go from 0 to 60 mph)")
    makeDotChart(getCol(table,9), "step-3-msrp.pdf", "MSRP of All Cars", "MSRP (in US Dollars)")

    # make bar charts for mpg based on two bin approaches
    # 1st approach uses US DOE rating
    ratings = [13, 14, 16, 19, 23, 26, 30, 36, 44]
    mpgRatings = getMpgRatings(getCol(table,0))
    makeBarChart(mpgRatings, "step-4-approach-1.pdf", "Fuel Economy Rating Frequency", "US Dept of Energy Fuel Economy Rating", 1, ratings)
    # 2nd approach makes equal width bins
    mpgRightBordersLst = getRightBordersList(getCol(table,0), 5)
    mpgRatingsWithBorders = getMPGRatingsWithBorders(getCol(table,0), mpgRightBordersLst)
    makeBarChart(mpgRatingsWithBorders, "step-4-approach-2.pdf", "MPG Frequency", str(5) + "Bin MPG Rating", 2, mpgRightBordersLst)

    # Make histograms for continuous attributes
    makeHistogram(getCol(table,0), "step-5-mpg.pdf", "Miles per Gallon of All Cars", "Miles per gallon")
    makeHistogram(getCol(table,2), "step-5-displacement.pdf", "Displacement of All Cars", "Displacement (cubic inches)")
    makeHistogram(getCol(table,3), "step-5-horsepower.pdf", "Horsepower of All Cars", "Horsepower (1 = 550 foot lbs per second)")
    makeHistogram(getCol(table,4), "step-5-weight.pdf", "Weight of All Cars", "Weight (lbs)")
    makeHistogram(getCol(table,5), "step-5-acceleration.pdf", "Acceleration of All Cars", "Acceleration (seconds to go from 0 to 60 mph)")
    makeHistogram(getCol(table,9), "step-5-msrp.pdf", "MSRP of All Cars", "MSRP (in US Dollars)")

    # Make scatter plots for continuous attributes
    makeScatterPlot(table, 2, 0, "step-6-displacement.pdf", "Displacement vs MPG", "Displacement (cubic inches)", "Miles per gallon", None)
    makeScatterPlot(table, 3, 0, "step-6-horsepower.pdf", "Horsepower vs MPG", "Horsepower (1 = 550 foot lbs per second)", "Miles per gallon", None)
    makeScatterPlot(table, 4, 0, "step-6-weight.pdf", "Weight vs MPG", "Weight (lbs)", "Miles per gallon", None)
    makeScatterPlot(table, 5, 0, "step-6-acceleration.pdf", "Acceleration vs MPG", "Acceleration (seconds to go from 0 to 60 mph)", "Miles per gallon", None)
    makeScatterPlot(table, 9, 0, "step-6-msrp.pdf", "MSRP vs MPG", "MSRP (in US Dollars)", "Miles per gallon", None)

    # Make scatter plots for continuous attributes 
    # with regression lines 
    # annotating correlation coefficient and covalence values
    regressionVals = leastSquares(table, 2, 0)
    makeScatterPlot(table, 2, 0, "step-7-displacement.pdf", "Displacement vs MPG", "Displacement (cubic inches)", "Miles per gallon", regressionVals)
    regressionVals = leastSquares(table, 3, 0)
    makeScatterPlot(table, 3, 0, "step-7-horsepower.pdf", "Horsepower vs MPG", "Horsepower (1 = 550 foot lbs per second)", "Miles per gallon", regressionVals)
    regressionVals = leastSquares(table, 4, 0)
    makeScatterPlot(table, 4, 0, "step-7-weight.pdf", "Weight vs MPG", "Weight (lbs)", "Miles per gallon", regressionVals)
    regressionVals = leastSquares(table, 5, 0)
    makeScatterPlot(table, 5, 0, "step-7-acceleration.pdf", "Acceleration vs MPG", "Acceleration (seconds to go from 0 to 60 mph)", "Miles per gallon", regressionVals)
    regressionVals = leastSquares(table, 9, 0)
    makeScatterPlot(table, 9, 0, "step-7-msrp.pdf", "MSRP vs MPG", "MSRP (in US Dollars)", "Miles per gallon", regressionVals)

    regressionVals = leastSquares(table, 2, 4)
    makeScatterPlot(table, 2, 4, "step-7-displacementvsweight.pdf", "Displacement vs Weight", "Displacement (cubic inches)", "Weight (lbs)", regressionVals)
    
    # Make box plot describing MPG by year
    makeBoxPlot(table)

    # Make frequency diagram of 
    # the number of cars from each country of origin 
    # separated out by model year
    makeMultFreq(table)

    pyplot.close()
    

if __name__ == '__main__':
    main()
