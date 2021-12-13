from collections import Counter
import csv
import string
import nltk
import pandas as pd
import random
import math
import datetime

tunigrams=[]
tbigrams=[]
   
def countWOrdUni(strg):    
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    exclude = set(string.punctuation)
    text = strg.translate(replace_punctuation).lower() 
    unig = Counter(nltk.ngrams(text.split(),1))
    dic={''.join(ch for ch in key if ch not in exclude):value for key, value in unig.items()}
    unigm=[''.join(ch for ch in key if ch not in exclude) for key, value in unig.items()]
    return dic,unigm

def countWOrdBi(strg):    
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    exclude = set(string.punctuation)
    text = strg.translate(replace_punctuation).lower() 
    bigmC = Counter(nltk.bigrams(text.split()))
    dic={' '.join(ch for ch in key if ch not in exclude):value for key, value in bigmC.items()}
    bigm=[' '.join(ch for ch in key if ch not in exclude) for key, value in bigmC.items()]
    return dic,bigm

def WordPro(P_word,N_word,thrs):
    uniProb={}    
    for tg, n_count in N_word.items():        
        if tg in P_word.keys() and P_word[tg]>=thrs:
            p_count=P_word[tg]
            c2 = float(p_count)
            p = (c2+1)/(c2+float(n_count)+2)
            uniProb[tg ]=(p,p_count,n_count)
           #print (uniProb[tg ],p)
    return uniProb

def classifier( thrs, unigrams, bigrams, U_list,B_list):
    WordProbMult = 1
    
    Positive_Bigams = []
    Positive_Unigams = []
    for unigram in unigrams:
        if unigram in U_list.keys():
            pr = float(U_list[unigram][0])
            if pr >= thrs:
                WordProbMult = WordProbMult * (1 - pr)
                Positive_Unigams.append(unigram)

    for bigram in bigrams:
        if bigram in B_list.keys():
            pr = float(B_list[bigram][0])    
            if pr >= thrs :
                WordProbMult = WordProbMult * (1 - pr)
                Positive_Bigams.append(bigram)
            else:
                Positive_Bigams.append(bigram)
    NoisyOR = 1 - WordProbMult
    return NoisyOR, Positive_Unigams, Positive_Bigams

def classify(testDataDf,outputCsv,TrnSize,PosSize,NegSize,uniPro, biPro):
    clsThrss=[0.0, .1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,0.97,0.99]
    wordProThrss=[0.0,.05, .1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,0.97,0.99]
    TN_WZ=0
    FN_WZ=0
    TP_WZ=0
    FP_WZ=0 
    record=[]
    
    for  ClsThrs in clsThrss:
        start = datetime.datetime.now()
        for WrdProThrs in wordProThrss:
            for row2 in range (0, testDataDf.shape[0]):                            
                C1_S, C1_U, C1_B= classifier( WrdProThrs, tunigrams[row2], tbigrams[row2],uniPro, biPro)                   
                if  C1_S<ClsThrs:
                    if testDataDf.iloc[row2][2]==0:
                        TN_WZ+=1
                    else:
                        FN_WZ+=1                
                    ##writer.writerow({'ClsThrs':ClsThrs,'WrdProThrs':WrdProThrs,'TrnSize':TrnSize,'PosSize':PosSize,'NegSize':NegSize,'TweetID':testDataDf.iloc[row2][0], 'Predicted_Class': 0,'Actual_Class':testDataDf.iloc[row2][2],'Classification_Score': C1_S,'Unigrams':C1_U, 'Bigrams':C1_B,'tweet':testDataDf.iloc[row2][1]})
    
                else: 
                    if testDataDf.iloc[row2][2]==1:
                        TP_WZ+=1
                    else:
                        FP_WZ+=1 
                    ##writer.writerow({'ClsThrs':ClsThrs,'WrdProThrs':WrdProThrs,'TrnSize':TrnSize,'PosSize':PosSize,'NegSize':NegSize,'TweetID': testDataDf.iloc[row2][0], 'Predicted_Class': 1,'Actual_Class':testDataDf.iloc[row2][2],'Classification_Score': C1_S,'Unigrams':C1_U, 'Bigrams':C1_B,'tweet':testDataDf.iloc[row2][1]})
    
            # print(T1,T2,'matrics',[100*(TP_WZ)/(TP_WZ+FP_WZ),100*(TP_WZ)/(TP_WZ+FN_WZ),100*(TN_WZ+TP_WZ)/(TN_WZ+TP_WZ+FN_WZ+FP_WZ)])
            preci=100*(TP_WZ)/(TP_WZ+FP_WZ+.00001)
            rec=100*(TP_WZ)/(TP_WZ+FN_WZ+.00001)
            accuracy=100*(TN_WZ+TP_WZ)/(TN_WZ+TP_WZ+FN_WZ+FP_WZ+.0001)
            print(TrnSize,NegSize,PosSize,ClsThrs,WrdProThrs,accuracy)
            FScore=2*preci*rec/(preci+rec+0.0000001)        
            TN_WZ=0
            FN_WZ=0
            TP_WZ=0
            FP_WZ=0
            record.append(str(TrnSize)+','+str(PosSize)+','+str(ClsThrs)+','+str(WrdProThrs)+','+str(FScore)+','+str(preci)+','+str(rec)+','+str(accuracy))
            #with open("EvluationMatric.csv", 'a',encoding="utf8") as files:
                #next(files)
                #record=str(TrnSize)+','+str(PosSize)+','+str(ClsThrs)+','+str(WrdProThrs)+','+str(FScore)+','+str(preci)+','+str(rec)+','+str(accuracy)
                #files.write(record)
                #files.write('\n')
        end = datetime.datetime.now()
        print('Time:',end-start)
    with open("EvluationMatric.csv", 'a',encoding="utf8") as files:
        #next(files)
        for row in record:
            files.write(row)
            files.write('\n')
            
def Robust_Classifier(csvDataSet,outputCsv,labelFieldName,textFieldName,Dsplit,Psplit,Nsplit):
    inputcsv=pd.read_csv(csvDataSet,encoding="utf-8")
    inputcsv['work_zone'].astype('int')
    inputdf=inputcsv.sample(frac =1,random_state=7)
    trainTestSplit=math.floor(inputdf.shape[0]*.8)
    dfTestData=inputdf.iloc[trainTestSplit:]                             
    for row2 in range (0, dfTestData.shape[0]):
        _,tug = countWOrdUni(dfTestData.iloc[row2][1])
        _,tbg=countWOrdBi(dfTestData.iloc[row2][1])
        tunigrams.append(tug) 
        tbigrams.append(tbg)
            
    df=inputdf.iloc[:trainTestSplit]
    
    #dataset_split=[i for i in range(int(df.shape[0]/2),df.shape[0],int((df.shape[0]-df.shape[0]/2)/(Dsplit-2)))]     
    #if df.shape[0] not in dataset_split:
        #dataset_split=dataset_split+[df.shape[0]]
    dataset_split=[i for i in range(100,2000,100)] 
    print(dataset_split)
    
    for i in dataset_split:    
        dftrain=df.iloc[:i]
        
        dfN=dftrain[dftrain[labelFieldName]==0]
        dfP=dftrain[dftrain[labelFieldName]==1]
        

        ##PText= ' '.join ( i for i in dfP[textFieldName])
        ##NText= ' '.join(i for i in dfN[textFieldName])
        Plength=dfP.shape[0] 
        Nlength=dfN.shape[0]
        #print( Nlength,Plength)
        random.seed(1) 
        NRandom=random.sample(range(int(Nlength/2),Nlength),Nsplit)
        if Nlength not in NRandom:
            NRandom=NRandom+[Nlength]
        PRandom=random.sample(range(int(Plength/2),Plength),Psplit)
        if Plength not in PRandom:
            PRandom=PRandom+[Plength]  
        print(NRandom,PRandom )   
        for nRan in NRandom:
            for pRan in PRandom:
                #print(nRan,pRan )
                PText= ' '.join ( i for i in dfP[textFieldName].iloc[:pRan])
                NText= ' '.join(i for i in dfN[textFieldName].iloc[:nRan]) 
                start = datetime.datetime.now()               
                P_word_uni,_=countWOrdUni(PText)
                N_word_uni,_=countWOrdUni(NText) 
                print('pretime',datetime.datetime.now()-start)                   
                P_word_bi,_=countWOrdBi(PText)
                N_word_bi,_=countWOrdBi(NText)                
                uniPROB=WordPro(P_word_uni,N_word_uni,1) 
                biPROB=WordPro(P_word_bi,N_word_bi,1)  
                classify(dfTestData,outputCsv,pRan+nRan,pRan,nRan,uniPROB, biPROB) 
    
inputdata='WZ_COMBINED_FINAL_TRAIN_TEST.csv'  
outputcsv="predicted_result.csv"

#clsThrss=[0.0]
#wordProThrss=[0.0]
def Model(inputdata,outputcsv):
    with open("EvluationMatric.csv", 'w',encoding="utf8") as f:
        fieldname = ['TrnSize','PosSize','ClsThrs','WrdProThrs','FScore','preci', 'rec','accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldname)
        writer.writeheader()  
        
    with open(outputcsv, 'w', newline='',encoding="utf8") as f:
        fieldname = ['ClsThrs','WrdProThrs','TrnSize','PosSize','NegSize','TweetID', 'Predicted_Class','Actual_Class', 'Classification_Score','Unigrams','Bigrams','tweet']
        writer = csv.DictWriter(f, fieldnames=fieldname)
        writer.writeheader()        

    Robust_Classifier(inputdata,outputcsv,"work_zone","text",10,1,1)  
    print(" ....Successfully Completed..")
Model(inputdata,outputcsv)        