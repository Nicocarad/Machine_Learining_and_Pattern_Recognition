import sys

 # define a function to compute the average
def compute_avg_score(lscores):                 # the function receives a list as input
    return sum(sorted(lscores)[1:-1])           # sum of the element of the list ignoring the first and the last so we take a slice after a sorting

# define a competitor class
class Competitor:
    def __init__(self, name, surname, country, scores):  #init is the special method invoked each time a class is instantiated, self is the new object, (name,surname,country,scores) are the attributes 
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
        self.avg_score = compute_avg_score(self.scores)


if __name__ == '__main__': #

    l_bestCompetitors = [] # define a empty list 
    hCountryScores = {} # define an empty dictionary
    with open(sys.argv[1]) as f:  # open the file containing data ( using the with sintax )
        for line in f: # iterates over the rows of the file
            name, surname, country = line.split()[0:3]  # split() with no arguments, split considering the space " " as separator
            scores = line.split()[3:]     # retreive the scorse considering the elements from index 3 and on
            scores = [float(i) for i in scores] # built a list of scores and cast to float
            comp = Competitor( # create a new competitor
                name,
                surname,
                country,
                scores)
            l_bestCompetitors.append(comp) # after creating the new competitor, add it to the list 
            if len(l_bestCompetitors) >= 4:
                l_bestCompetitors = sorted(l_bestCompetitors, key = lambda i: i.avg_score)[::-1][0:3] # sort the list of best competitors according to the key (the key is extracted with a lambda function and it is the average score)
                                                                                                      # then reverse the list with [::-1] and take only the first three competitors
            if comp.country not in hCountryScores:       
                hCountryScores[comp.country] = 0
            hCountryScores[comp.country] += comp.avg_score

    if len(hCountryScores) == 0:
        print('No competitors')
        sys.exit(0)

    best_country = None
    for count in hCountryScores:
        if best_country is None or hCountryScores[count] > hCountryScores[best_country]:
            best_country = count

    print('Final ranking:')
    for pos, comp in enumerate(l_bestCompetitors):
        print('%d: %s %s - Score: %.1f' % (pos+1, comp.name, comp.surname, comp.avg_score))
    print()
    print('Best Country:')
    print("%s - Total score: %.1f" % (best_country, hCountryScores[best_country]))
