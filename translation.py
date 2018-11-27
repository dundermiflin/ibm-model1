import string
import numpy as np
from timeit import default_timer as timer

start= timer()

epochs=10
table= str.maketrans({key:None for key in string.punctuation})
flatten= lambda l: [word for row in l for word in row]

def prob_e_f(g,e,tt):
    e_l= len(e)
    g_l= len(g)

    ans=1
    for gw in g:
        sum=0
        for ew in e:
            sum+=tt[(gw,ew)]
        ans*=sum
    ans*=1/(e_l**g_l)
    return ans

#Reading datasets
eFile= open("English.txt",'r')
eText= eFile.read().lower()
#Removing punctuation
eText= eText.translate(table).split("\n")
num_sent= len(eText)
eMat=[]
for sent in eText:
    row= []
    for word in sent.split(" "):
        if word!="":
            row.append(word)
    eMat.append(row)

#Creating language vocabulary and term-frequency
eTerms= flatten(eMat)
eVocab= list(set(eTerms))
eTermFreq= {term:eTerms.count(term) for term in eVocab}

gFile= open("German.txt",'r')
gText= gFile.read().lower()
#Removing punctuation
gText= gText.translate(table).split("\n")
gMat=[]
for sent in gText:
    row= []
    #Filtering empty strings
    for word in sent.split(" "):
        #Filtering empty strings
        if word!="":
            row.append(word)
    gMat.append(row)

#Creating language vocabulary and term-frequency
gTerms= flatten(gMat)
gVocab= list(set(gTerms))
gTermFreq= {term:gTerms.count(term) for term in gVocab}

parallelCorpus= [(gMat[i],eMat[i]) for i in range(num_sent)]

#Initialising translation table
tt= {}
for gw in gVocab:
    for ew in eVocab:
        tt[(gw,ew)]= 1/(len(eVocab))

#Training model
for e in range(epochs):
    target_total={}
    count={}
    source_total={}

    #Resetting the values
    for ew in eVocab:
        target_total[ew]=0
        for gw in gVocab:
            count[(gw,ew)]=0

    #Calculating the denominators/normalizing values
    for sp in parallelCorpus:
        for gw in sp[0]:
            source_total[gw]=0
            for ew in sp[1]:
                source_total[gw] += tt[(gw,ew)]
        
        #Calculating mapped translation values
        #Calculating denominators for probability
        for gw in sp[0]:
            for ew in sp[1]:
                count[(gw,ew)]+= tt[(gw,ew)]/source_total[gw]
                target_total[ew]+= tt[(gw,ew)]/source_total[gw]
    
    #Recalculating probability values 
    for ew in eVocab:
        for gw in gVocab:
            tt[(gw,ew)]= count[(gw,ew)]/target_total[ew]

#Extracting alignments from probablity scores
alignments= []   
for i in range(len(gMat)):
    print("Alignment for sentence {0}:".format(i+1))
    align=[]
    for word in gMat[i]:
            scores= [tt[(word,ew)] for ew in eMat[i]]
            ind= scores.index(max(scores))
            align.append((gMat[i].index(word),ind))
    alignments.append(align)
    print(align)

end= timer()
print()
print("The execution of the program took {0} seconds".format(round(end-start,3)))