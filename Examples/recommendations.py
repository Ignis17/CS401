'''
Joel Turbi
Dr. Zavala
CS401 - CS/Telecommunication
Recommender System
'''
from math import sqrt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections

# Set your work directory to where the data is located
# pd.read_csv(r"C:/Users/Ignis17/WSL/CS401/Examples/ml-100k/")
os.chdir("/mnt/c/Users/Ignis17/WSL/CS401/Examples/ml-100k/")

# A dictionary of movie critics and their ratings of a small set of movies
# critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
#          'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
#          'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5,
#          'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
#          'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
#          'Superman Returns': 3.5, 'The Night Listener': 4.0}, 'Claudia Puig': {'Snakes on a Plane': 3.5,
#          'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0,
#          'You, Me and Dupree': 2.5}, 'Mick LaSalle': {'Lady in the Water': 3.0,
#          'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,
#          'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
#          'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
#          'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
#          'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}


def generateData():
    #  Empty dictionary
    critics = collections.defaultdict(dict)
    # column headers for the dataset
    data_cols = ['user id','movie id','rating','timestamp']
    item_cols = ['movie id','movie title','release date',
    'video release date','IMDb URL','unknown','Action',
    'Adventure','Animation','Childrens','Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror',
    'Musical','Mystery','Romance ','Sci-Fi','Thriller',
    'War' ,'Western']
    user_cols = ['user id','age','gender','occupation','zip code']

    # importing the data files onto dataframes:
    users = pd.read_csv('u.user', sep='|',names=user_cols, encoding='latin-1')
    item = pd.read_csv('u.item', sep='|',names=item_cols, encoding='latin-1')
    data = pd.read_csv('u.data', sep='\t',names=data_cols, encoding='latin-1')

    # Create a merged dataframe:
    df = pd.merge(pd.merge(item, data), users)

    # Store data individually:
    userdata = list(df.get("user id", default=None))
    titles = list(df.get("movie title", default=None))
    ratings = list(df.get("rating", default=None))
    # sliced number of inputs: given that the total number of movies reviewed is 100k
    # and
    us = userdata[0:1000]
    ti = titles[0:1000]
    ra = ratings[0:1000]
    for i in us:
        for x in ti:
            for y in ra:
                critics[str(i)][x]=y

    return dict(critics)

# Returns a distanced-based similarity score for person1 and person2
def sim_distance(prefs, person1, person2):
    # Get the list of shared_items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    # if they have not ratings in common, return 0
    if len(si) == 0: return 0

    # Add up the squares of all differences
    sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item],2)for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sum_of_squares)

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item]=1
    # Find the number of elements
    n=len(si)
    # if they are no ratings in common, return 0
    if n==0: return 0
    # Add up all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    # Sum up the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
    # Sum up the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    # Calculate Pearson score
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0

    r=num/den

    return r

# Returns the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
# scores everyone against a given person and finds the closest matches.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
    scores=[(similarity(prefs,person,other),other)
                    for other in prefs if other!=person]
    # Sort the list so the highest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        # don't compare me to myself
        if other==person: continue
        sim=similarity(prefs,person,other)

        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
             # only score movies I haven't seen yet
             if item not in prefs[person] or prefs[person][item]==0:
                 # Similarity * Score
                 totals.setdefault(item,0)
                 totals[item]+=prefs[other][item]*sim
                 # Sum of similarities
                 simSums.setdefault(item,0)
                 simSums[item]+=sim
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items( )]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

# Function to swap people and the items.
def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})

            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

# Test cases:

# result = sim_distance(critics,'Lisa Rose','Gene Seymour')
# result = sim_pearson(critics,'Lisa Rose','Gene Seymour')
# result = topMatches(critics,'Toby',n=3)
# result = getRecommendations(critics,'Toby')
# result = getRecommendations(critics,'Toby',similarity=sim_distance)
# print(result)

# movies = transformPrefs(critics)
# result = topMatches(movies,'Superman Returns')
# Get recommended critics for a movie.
# result = getRecommendations(movies,'Just My Luck')
# result = topMatches(critics, "Jack Matthews", similarity=sim_distance)
# result = getRecommendations(critics,"Lisa Rose", similarity = sim_distance)
# result = topMatches(critics, 'Lisa Rose')
# result = topMatches(movies, "Superman Returns")
# print(result)


# print(generateData())
critics = generateData()
# result = getRecommendations(critics, "144")
result = topMatches(critics,"308", n=3)
print(result)
