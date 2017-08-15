import json
import numpy as np


average_time_per_page = 20
average_time_on_website = 300
nbPageMin = 5
len_seq_min_acq = 3
taux_acq = 0.5
size = 100000
nbPage = 15
visite = 0


#Define acquisition or not
def acqOrNot():
    acq_p = np.random.random_sample()
    if acq_p <= taux_acq:
        return True
    return False

def subSeqInSeq(seq,tab):
    for i in seq:
        if(i not in tab):
            return False
    return True

def getPagesVisited(tab):
    pagesVisited = []
    for p in tab:
        pagesVisited.append(p[0])
    return pagesVisited

def getPagesVisitedWithoutTime(tab):
    pagesVisited = []
    for p in tab:
        pagesVisited.append(p)
    return pagesVisited

def emplaceSeqinTab(seq,tab,randomly=False):
    if(randomly):
        index = np.random.randint(1,len(tab),len(seq))
    else:
        start_index = np.random.randint(1,len(tab) - len(seq))
        index = []
        l = 0
        while(l < len(seq)):
            index.append(start_index + l)
            l+=1
    for j in range(len(seq)):
        tab[index[j]][0] = seq[j]

def emplaceSeqinTabWithoutTime(seq,tab,randomly=False):
    if(randomly):
        index = np.random.randint(1,len(tab),len(seq))
    else:
        start_index = np.random.randint(1,len(tab) - len(seq))
        index = []
        l = 0
        while(l < len(seq)):
            index.append(start_index + l)
            l+=1
    for j in range(len(seq)):
        tab[index[j]] = seq[j]

def dataCheck(pagesData,labels,distinct=True):
    e = 0
    a = 0
    k = 0
    while k  < len(pagesData):
        acq = subSeqInSeq(seq_page_acq, pagesData[k])
        if (not acq and labels[k] == 1):
            print("Warning: wrong data inserted")
            e+=1
        if(acq and labels[k] == 0):
            if(distinct):
                l = 0
                while(l < len(data[k])):
                    if(data[k][l][0] in seq_page_acq):
                        data[k][l][0] = np.random.randint(1,nbPage)
                    l+=1
                #print("Warning: misleading data\n"+ str(pagesData[k]) +"\nreplaced by\n" + str(getPages(k)))
        if(acq):
            a+=1
        k+=1
    print("Taux acq data:" + str(float(a)/size))
    print("Error insert rate:" + str(float(e)/size))


def dataCheckWithoutTime(pagesData,labels,distinct=True):
    e = 0
    a = 0
    k = 0
    while k  < len(pagesData):
        acq = subSeqInSeq(seq_page_acq, pagesData[k])
        if (not acq and labels[k] == 1):
            print("Warning: wrong data inserted")
            e+=1
        if(acq and labels[k] == 0):
            if(distinct):
                l = 0
                while(l < len(data[k])):
                    if(data[k][l] in seq_page_acq):
                        data[k][l] = np.random.randint(1,nbPage)
                    l+=1
                #print("Warning: misleading data\n"+ str(pagesData[k]) +"\nreplaced by\n" + str(getPages(k)))
        if(acq):
            a+=1
        k+=1
    print("Taux acq data:" + str(float(a)/size))
    print("Error insert rate:" + str(float(e)/size))

def saveData():
    with open('data'+str(size)+'.json', 'w') as outfile:
        json.dump(data, outfile)

    with open('label'+str(size)+'.json', 'w') as outfile:
        json.dump(label, outfile)

    print("Data have been saved")

def getPages(k):
    p_ = []
    for p in data[k]:
        p_.append(p[0])
    return p_

# init output
data = []
label = []
pages = []

# init

seq_page_acq = np.random.randint(0,nbPage,len_seq_min_acq)
max_temps_total = 0
temps_total_per_page = {}
min_seq = 99999999
for i in range(nbPage):
    temps_total_per_page[i] = 0

def preTreatmentData(data,max_size=None):
    print(True)
    i = 0
    while (i < len(data)):
        k = 1
        while (k < len(data[i])):
            data[i][k][1] = data[i][k][1]/float(temps_total_per_page[data[i][k][0]])
            k+=1
        data[i][0][1] = data[i][0][1]/float(max_temps_total)
        tab = [data[i][0]]
        tab += data[i][(len(data[i])-min_seq):]
        data[i] = tab
        i+=1

def preTreatmentDataWithoutTime(data):
    print(True)
    i = 0
    while (i < len(data)):
        tab = data[i][(len(data[i]) - min_seq):]
        data[i] = tab
        i += 1

# loop on number of lines wanted

while visite < size :
    acq = acqOrNot()
    # generate a normal random time on website
    time_on_website = np.random.normal(average_time_on_website, 50)
    # check if time on website > 0
    if time_on_website < 0:
        time_on_website = 100.25

    # init line
    #track = [[nbPage+1,time_on_website]]
    track = []

    # init time step
    t = 0
    nbPageVisited = 0
    pageVisited = []
    while t < time_on_website or nbPageVisited < nbPageMin:
        # generate a random page number
        page = np.random.randint(0,nbPage)
        pageVisited.append(page)
        # generate a normal random time step
        time_step = np.random.normal(average_time_per_page, 5)
        # check if time step > 0
        while( time_step < 0 ):
            time_step = np.random.normal(average_time_per_page, 5)
        # add page to track
        #track.append([page, time_step])
        track.append([page])
        temps_total_per_page[page] = max(time_step,temps_total_per_page[page])
        t += time_step
        nbPageVisited += 1

    if(acq):
        if(not subSeqInSeq(seq_page_acq, pageVisited)):
            emplaceSeqinTab(seq_page_acq,track)
        label.append(1)
    else:
        label.append(0)

    max_temps_total = max(time_on_website,max_temps_total)
    # add line to output

    data.append(track)
    pages.append(getPagesVisitedWithoutTime(track))
    min_seq = min(min_seq,len(track))
    visite += 1

preTreatmentDataWithoutTime(data)
dataCheck(pages,label)
r = float(np.sum(label))/len(label)
print("Acq rate after correction: " + str(r))
print("Acquisition Seq: "+str(seq_page_acq))
print("Seqence minimum:" + str(min_seq))

if(np.abs(r - taux_acq) <= 0.01):
    saveData()

