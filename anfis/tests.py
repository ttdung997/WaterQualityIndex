import anfis
# import membership.mfDerivs
import numpy as np

def partial_dMF(x, mf_definition, partial_parameter):
    """Calculates the partial derivative of a membership function at a point x.



    Parameters
    ------


    Returns
    ------

    """
    mf_name = mf_definition[0]

    if mf_name == 'gaussmf':

        sigma = mf_definition[1]['sigma']
        mean = mf_definition[1]['mean']

        if partial_parameter == 'sigma':
            result = (2./sigma**3) * np.exp(-(((x-mean)**2)/(sigma)**2))*(x-mean)**2
        elif partial_parameter == 'mean':
            result = (2./sigma**2) * np.exp(-(((x-mean)**2)/(sigma)**2))*(x-mean)

    elif mf_name == 'gbellmf':

        a = mf_definition[1]['a']
        b = mf_definition[1]['b']
        c = mf_definition[1]['c']

        if partial_parameter == 'a':
            result = (2. * b * np.power((c-x),2) * np.power(np.absolute((c-x)/a), ((2 * b) - 2))) / \
                (np.power(a, 3) * np.power((np.power(np.absolute((c-x)/a),(2*b)) + 1), 2))
        elif partial_parameter == 'b':
            result = -1 * (2 * np.power(np.absolute((c-x)/a), (2 * b)) * np.log(np.absolute((c-x)/a))) / \
                (np.power((np.power(np.absolute((c-x)/a), (2 * b)) + 1), 2))
        elif partial_parameter == 'c':
            result = (2. * b * (c-x) * np.power(np.absolute((c-x)/a), ((2 * b) - 2))) / \
                (np.power(a, 2) * np.power((np.power(np.absolute((c-x)/a),(2*b)) + 1), 2))

    elif mf_name == 'sigmf':

        b = mf_definition[1]['b']
        c = mf_definition[1]['c']

        if partial_parameter == 'b':
            result = -1 * (c * np.exp(c * (b + x))) / \
                np.power((np.exp(b*c) + np.exp(c*x)), 2)
        elif partial_parameter == 'c':
            result = ((x - b) * np.exp(c * (x - b))) / \
                np.power((np.exp(c * (x - c))) + 1, 2)


    return result

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:41:58 2014

@author: tim.meggs
"""

from skfuzzy import gaussmf, gbellmf, sigmf

class MemFuncs:
    # 'Common base class for all employees'
    funcDict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}


    def __init__(self, MFList):
        self.MFList = MFList

    def evaluateMF(self, rowInput):
        if len(rowInput) != len(self.MFList):
            print("Number of variables does not match number of rule sets")

        return [[self.funcDict[self.MFList[i][k][0]](rowInput[i],**self.MFList[i][k][1]) for k in range(len(self.MFList[i]))] for i in range(len(rowInput))]

import numpy


dictrictArr = ['Gayang' ,'GoDeokcheon Stream', 'Noryangjin', 'Dorimcheon Stream',
 'Mokgamcheon Stream' ,'Seongnaecheon Stream' ,'Seongbukcheon Stream',
 'Ahnyangcheon Stream 4' ,'Ahnyangcheon Stream 5', 'Amsa',
 'Yangjaecheon Stream', 'Uicheon Stream', 'Jeongneungcheon Stream',
 'Jungnangcheon Stream 1A', 'Jungnangcheon Stream 2',
 'Jungnangcheon Stream 3', 'Jungnangcheon Stream 4',
 'Cheonggyecheon Stream 1', 'Cheonggyecheon Stream 2',
 'Cheonggyecheon Stream 3', 'Tancheon Stream 5' ,'Hongjecheon Stream', 'Guui',
 'Ttukdo' ,'Bogwang', 'Yeongdeungpo', 'Jamsil', 'Mangwolcheon Stream',
 'Gamicheon Stream', 'Mokgamcheon Stream-1', 'Ahnyangcheon Stream 3-2',
 'Choicheon Stream', 'Mesotrophic']

dictrictArr = ['Mesotrophic']


for dictrict in dictrictArr:
    # ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
    ts = numpy.loadtxt("data/"+dictrict+".txt", usecols=[0,1,2,3,4])
    X = ts[:,0:4]
    Y = ts[:,4]

    print(X.shape)

    print(Y.shape)
    mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
    [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]],
    [['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':3.,'sigma':4.}]],
    [['gaussmf',{'mean':3.,'sigma':4.}],['gaussmf',{'mean':4.,'sigma':5.}]]]

    print(len(mf))
    mfc = MemFuncs(mf)

    anf = anfis.ANFIS(X, Y, mfc)
    anf.trainHybridJangOffLine(epochs=20)
    print(round(anf.consequents[-1][0],6))
    print(round(anf.consequents[-2][0],6))
    print(round(anf.fittedValues[9][0],6))
    if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
    	print('test is good')
    # anf.plotErrors()
    print("_________________________________________")
    print(dictrict)

    anf.plotResults()