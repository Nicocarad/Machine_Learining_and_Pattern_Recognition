import sys

#defining a class BusRecord with some fields
class BusRecord:
    def __init__(self, busId, lineId, x, y, t):
        self.busId = busId
        self.lineId = lineId
        self.x = x
        self.y = y
        self.t = t

 # open the file and store the information in a list       
def loadAllRecords(fName):
    
    try:
        lRecords = [] # create an empty list 
        with open(fName) as f:
            for line in f:
                busId, lineId, x, y, t = line.split() #split the line using " " as delimiter and save each value in the corresponding variable
                record = BusRecord(busId, lineId, int(x), int(y), int(t))
                lRecords.append(record) # populate the list with elements of class BusRecord
        return lRecords
    except:
        raise # If we do not provide an exception, the current exception is propagated

# define a function to calculate the euclidean distance
# r1 and r2 are records
# the function, takes the x and y coordinates to compute the distance
def euclidean_distance(r1, r2):
    return ((r1.x-r2.x)**2 + (r1.y-r2.y)**2)**0.5


def computeBusDistanceTime(lRecords, busId):
    busRecords = sorted([i for i in lRecords if i.busId == busId], key = lambda x: x.t) # assign to busRecord, the object of class Bus associated to the busID passed as parameter
    # sort the object, according to a key: x coordinate
    print(busRecords) #debug stdout 
    if len(busRecords) == 0:
        return None, None
    totDist = 0.0
    for prev_record, curr_record in zip(busRecords[:-1], busRecords[1:]): # create an iterable with zip, considering only object of busRecord excluding the last element and the first element
        totDist += euclidean_distance(curr_record, prev_record) # compute the euclidean distance and sum all the distances
    totTime = busRecords[-1].t - busRecords[0].t # compute the total time 
    return totDist, totTime

def computeLineAvgSpeed(lRecords, lineId):
    
    lRecordsFiltered = [i for i in lRecords if i.lineId == lineId] # store in lRecordsFiltered, only records satisfying the constraint
    busSet = set([i.busId for i in lRecordsFiltered]) # create a set containing the busId 
    if len(busSet) == 0:
        return 0.0
    totDist = 0.0
    totTime = 0.0
    for busId in busSet:
        d, t = computeBusDistanceTime(lRecordsFiltered, busId)
        totDist += d
        totTime += t
    return totDist / totTime

if __name__ == '__main__': # it is like int main() in java or C

    lRecords = loadAllRecords(sys.argv[1])
    if sys.argv[2] == '-b':
        print('%s - Total Distance:' % sys.argv[3], computeBusDistanceTime(lRecords, sys.argv[3])[0])
    elif sys.argv[2] == '-l':
        print('%s - Avg Speed:' % sys.argv[3], computeLineAvgSpeed(lRecords, sys.argv[3]))
    else:
        raise KeyError()

