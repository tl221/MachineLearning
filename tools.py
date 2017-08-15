from numpy import asarray
from json import load

def loadJson(file):
    with open(file) as json_file:
        dat = load(json_file)
    return asarray(dat)

class DataSet:

    def __init__(self,data,labels,rate=0.8,visu=None):
        end = int(data.shape[0]*rate)
        self._dataTrain = data[:end]
        self._labelsTrain = labels[:end]
        self._dataTest = data[end:]
        self._labelsTest = labels[end:]
        self._currentPosition = 0
        self._nbFeatures = data.shape[1]
        if(visu):
            for i in range(visu):
                print("Data")
                print(self._dataTrain[i])

    def nextTrainBatch(self,jump):
        if(self._currentPosition + jump >= len(self._dataTrain)):
            jump = len(self._dataTrain) - self._currentPosition -1
        dat  = self._dataTrain[self._currentPosition:self._currentPosition+jump]
        lab  = self._labelsTrain[self._currentPosition:self._currentPosition+jump]
        self._currentPosition += jump
        return dat,lab

    def getDataTest(self):
        return self._dataTest,self._labelsTest

    def getDataTrain(self):
        return self._dataTrain,self._labelsTest

    def getNbFeatures(self):
        return self._nbFeatures

    def resetPosition(self):
        self._currentPosition = 0

if __name__ == "__main__":
    print("----- test data ------")
    da = loadJson('data100000.json')
    la = loadJson('label100000.json')
    data = DataSet(da, la, 0.8)



