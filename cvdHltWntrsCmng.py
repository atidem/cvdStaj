# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:23:07 2020

@author: atidem
"""

import pandas as pd 
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from statsmodels.tsa.ar_model import AR,ARResults
from statsmodels.tsa.arima_model import ARIMA,ARMA,ARIMAResults,ARMAResults
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
import matplotlib as mt
import statsmodels as st
import sklearn as sk
#%% Parameters

#%%
##!!! seperator can change ; to ,
df = pd.read_csv("Cov19-Tur.csv",index_col='date',sep=';')
df.index = pd.to_datetime(df.index,format="%d.%m.%Y")
df.index.freq = 'D'

dataLen = len(df)
#positivity
dataPos = df[df.Deaths>0]
dataPosLen = len(dataPos)
# size of predict(daily)
predDayCount = 30
# total range
totalIdx = pd.date_range(df.index[0],periods=dataLen+predDayCount,freq='D')

#df["Cases"][:pd.to_datetime("19.3.2020",format="%d.%m.%Y")]

#%% measure metrics
def mape(a,b): 
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean()

def mae(a,b):
    return mean_absolute_error(a,b)

def rmse(a,b):
    return sqrt(mean_squared_error(a,b))

#%% Holt Winters
"""
---Holt Winters---
alpha = smoothing_level
beta = smoothing_slope
gamma = smoothing_seasonal
phi = damping_slope
tren  = mul , add
seasonal = mul , add
seasonal period
damped   = True , False

Gonna add user interface 
"""

def holtWinters(data,alpha=None,beta=None,gamma=None,phi=None,tren=None,seasonal='add',period=None,damp=False):
    if (tren=='mul' or seasonal=='mul' ):
        dataPos = data[data>0]
        dataPosLen = len(dataPos)
    else:
        dataPos = data
        dataPosLen = len(data)
    
    pred = pd.DataFrame(index=totalIdx)
    model = ExponentialSmoothing(dataPos[:dataPosLen],trend=tren,seasonal=seasonal,seasonal_periods=period,damped=damp)
    pred["Fitted_Values"] = model.fit(smoothing_level=alpha,smoothing_slope=beta,smoothing_seasonal=gamma,damping_slope=phi).fittedvalues
    pred["Predicted_Values"] = pd.Series(model.predict(model.params,start=df.index[-1],end=totalIdx[-1]),index=totalIdx[dataLen-1:])
    return pred

## Holt Winters Prediction Section 
## default values (alpha=None,beta=None,gamma=None,phi=None,tren=None,seasonal='add',period=None,damp=False)
Case_mul_mul = holtWinters(data=df.Cases,alpha=0.25,beta=0.25,gamma=0,tren='mul',seasonal='mul',period=12,damp=True)
Case_mul_mul.rename(columns={"Fitted_Values":"Cases_hw_tes_mul-mul","Predicted_Values": "Cases_predict_hw_tes_mul"},inplace=True)

Case_add_add = holtWinters(data=df.Cases,alpha=0.9,beta=0.9,gamma=0,tren='add',seasonal='add',period=80,damp=False)
Case_add_add.rename(columns={"Fitted_Values":"Cases_hw_tes_add-add","Predicted_Values": "Cases_predict_hw_tes_add"},inplace=True)

Death_mul_mul = holtWinters(data=df.Deaths,alpha=0.9,beta=0.9,gamma=0,tren='mul',seasonal='mul',period=75,damp=True)
Death_mul_mul.rename(columns={"Fitted_Values":"Deaths_hw_tes_mul","Predicted_Values": "Deaths_predict_hw_tes_mul"},inplace=True) 

Death_add_add = holtWinters(data=df.Deaths,alpha=0.9,beta=0.9,gamma=0,tren='add',seasonal='add',period=80,damp=False)   
Death_add_add.rename(columns={"Fitted_Values":"Deaths_hw_tes_add","Predicted_Values": "Deaths_predict_hw_tes_add"},inplace=True) 

## merge prediction and main dataframe
finalDf = pd.concat([df,Case_mul_mul,Case_add_add,Death_mul_mul,Death_add_add],axis=1)
#%% AutoRegresive
"""
--- AR ---
maxlag = int

method = cmle,mle
// Conditional maximum likelihood using OLS ,Unconditional (exact) maximum likelihood.

lagOpt = aic,bic,hqic,t-stat 
//Akaike Information Criterion,Bayes Information Criterion,Hannan-Quinn Information Criterion,Based on last lag

trend = c,nc 
//constant, no constant

"""

def ar(data,maxlag=None,metod='cmle',lagOpt='t-stat',trend='nc',testRate=0.2):
    dataPos = data[data>0]
    dataPosLen = len(dataPos)
    
    splitIndex = int(dataPosLen*(1-testRate))
    train = dataPos[:splitIndex]
    test = dataPos[splitIndex:]
    model = AR(train)
    model = model.fit(maxlag=maxlag,method=metod,trend=trend,ic=lagOpt)
    pred = model.predict(start=totalIdx[dataLen-dataPosLen+splitIndex],end=totalIdx[-1])
    pred = pd.DataFrame(pred)
    maev = mae(test,pred[0][:len(test)])
    rmsev = rmse(test,pred[0][:len(test)])
    mapev = mape(pred[0][:len(test)],test)

    measure = {"mae":maev,"rmse":rmsev,"mape":mapev}
    
    return pred,measure

#  AR Prediction Section
## default values (maxlag=None,metod='cmle',lagOpt='t-stat',trend='nc',testRate=0.2)
Cases_Ar,Cases_Ar_Measure = ar(data=df['Cases'],trend='c')
Deaths_Ar,Death_Ar_Measure = ar(data=df['Deaths'],trend='c')
Cases_Ar.rename(columns={0:"Cases_predict_ar"},inplace=True)
Deaths_Ar.rename(columns={0:"Deaths_predict_ar"},inplace=True)

finalDf = pd.concat([finalDf,Cases_Ar,Deaths_Ar],axis=1)
#%% ARIMA

#there is something wrong this block
def arimaParametersFounder(data,startP=0,startQ=0,maxP=10,maxQ=10,testRate=0.2):
    dataPos = data[data>0]
    dataPosLen = len(dataPos)
    
    splitIndex = int(dataPosLen*(1-testRate))
    train = dataPos[:splitIndex]    
    model = auto_arima(train,start_p=startP,start_q=startQ,max_D=1,max_p=maxP,stationary=True,max_q=maxQ,seasonal=False,trace=True)
    return model   
    
def arima(data,p,d,q,testRate=0.2):
    dataPos = data[data>0]
    dataPosLen = len(dataPos)
    dataPos = pd.to_numeric(dataPos,downcast='float')
    
    splitIndex = int(dataPosLen*(1-testRate))
    train = dataPos[:splitIndex]
    test = dataPos[splitIndex:]

    model = ARIMA(train,order=(p,d,q)).fit()
    pred = model.predict(start=totalIdx[dataLen-dataPosLen+splitIndex],end=totalIdx[-1])
    pred = pd.DataFrame(pred)
    maev = mae(test,pred[0][:len(test)])
    rmsev = rmse(test,pred[0][:len(test)])
    mapev = mape(pred[0][:len(test)],test)
    
    measure = {"mae":maev,"rmse":rmsev,"mape":mapev}
    
    return pred,measure

#parameter search
#model=arimaParametersFounder(data=df['Cases'])
#model.summary()
##
#model=arimaParametersFounder(data=df['Deaths'])
#model.summary()
# ARIMA Prediction Section
Cases_Arima,Cases_Arima_Measure = arima(data=df['Cases'],p=2,d=0,q=1)
Deaths_Arima,Deaths_Arima_Measure = arima(data=df['Deaths'],p=1,d=0,q=0)
Cases_Arima.rename(columns={0:"Cases_predict_arima"},inplace=True)
Deaths_Arima.rename(columns={0:"Deaths_predict_arima"},inplace=True)

finalDf = pd.concat([finalDf,Cases_Arima,Deaths_Arima],axis=1)

#%% ARMA

def arma(data,p,q,testRate=0.2):
    dataPos = data[data>0]
    dataPosLen = len(dataPos)
    dataPos = pd.to_numeric(dataPos,downcast='float')
    
    splitIndex = int(dataPosLen*(1-testRate))
    train = dataPos[:splitIndex]
    test = dataPos[splitIndex:]

    model = ARMA(train,order=(p,q)).fit()
    pred = model.predict(start=totalIdx[dataLen-dataPosLen+splitIndex],end=totalIdx[-1])
    pred = pd.DataFrame(pred)
    maev = mae(test,pred[0][:len(test)])
    rmsev = rmse(test,pred[0][:len(test)])
    mapev = mape(pred[0][:len(test)],test)
    
    measure = {"mae":maev,"rmse":rmsev,"mape":mapev}
    
    return pred,measure

# ARMA Prediction Section
Cases_Arma,Cases_Arma_Measure = arma(data=df['Cases'],p=3,q=1)
Deaths_Arma,Deaths_Arma_Measure = arma(data=df['Deaths'],p=1,q=0)
Cases_Arma.rename(columns={0:"Cases_predict_arma"},inplace=True)
Deaths_Arma.rename(columns={0:"Deaths_predict_arma"},inplace=True)

finalDf = pd.concat([finalDf,Cases_Arma,Deaths_Arma],axis=1)
#%% measure
print("----Cases Measure----")
print("MAE TES MULT : "+str(mae(finalDf['Cases'][:dataLen],finalDf["Cases_hw_tes_mul-mul"][:dataLen])))
print("RMSE TES MULT : "+str(rmse(finalDf['Cases'][:dataLen],finalDf["Cases_hw_tes_mul-mul"][:dataLen])))
print("MAPE TES MULT : "+str(mape(finalDf['Cases'][:dataLen],finalDf["Cases_hw_tes_mul-mul"][:dataLen])))
print("..............................................................")
print("MAE TES ADD : "+str(mae(finalDf['Cases'][:dataLen],finalDf["Cases_hw_tes_add-add"][:dataLen])))
print("RMSE TES ADD : "+str(rmse(finalDf['Cases'][:dataLen],finalDf["Cases_hw_tes_add-add"][:dataLen])))
print("MAPE TES ADD : "+str(mape(finalDf['Cases'][:dataLen],finalDf["Cases_hw_tes_add-add"][:dataLen])))
print("..............................................................")
print("MAE AR : " + str(Cases_Ar_Measure["mae"]))
print("RMSE AR : " + str(Cases_Ar_Measure["rmse"]))
print("MAPE AR : " + str(Cases_Ar_Measure["mape"]))
print("..............................................................")
print("MAE ARMA : " + str(Cases_Arma_Measure["mae"]))
print("RMSE ARMA : " + str(Cases_Arma_Measure["rmse"]))
print("MAPE ARMA : " + str(Cases_Arma_Measure["mape"]))
print("..............................................................")
print("MAE ARIMA : " + str(Cases_Arima_Measure["mae"]))
print("RMSE ARIMA : " + str(Cases_Arima_Measure["rmse"]))
print("MAPE ARIMA : " + str(Cases_Arima_Measure["mape"]))
print("\n")

print("----Deaths Measure----")
print("MAE TES ADD : "+str(mae(finalDf['Deaths'][:dataLen],finalDf["Deaths_hw_tes_add"][:dataLen])))
print("RMSE TES ADD: "+str(rmse(finalDf['Deaths'][:dataLen],finalDf["Deaths_hw_tes_add"][:dataLen])))
print("MAPE TES ADD: "+str(mape(finalDf['Deaths'][:dataLen],finalDf["Deaths_hw_tes_add"][:dataLen])))
print("..............................................................")
print("MAE TES MULT : "+str(mae(dataPos['Deaths'][:dataPosLen],finalDf["Deaths_hw_tes_mul"][dataLen-dataPosLen:dataLen])))
print("RMSE TES MULT : "+str(rmse(dataPos['Deaths'][:dataPosLen],finalDf["Deaths_hw_tes_mul"][dataLen-dataPosLen:dataLen])))
print("MAPE TES MULT : "+str(mape(dataPos['Deaths'][:dataPosLen],finalDf["Deaths_hw_tes_mul"][dataLen-dataPosLen:dataLen])))
print("..............................................................")
print("MAE AR : " + str(Death_Ar_Measure["mae"]))
print("RMSE AR : " + str(Death_Ar_Measure["rmse"]))
print("MAPE AR : " + str(Death_Ar_Measure["mape"]))
print("..............................................................")
print("MAE ARMA : " + str(Deaths_Arma_Measure["mae"]))
print("RMSE ARMA : " + str(Deaths_Arma_Measure["rmse"]))
print("MAPE ARMA : " + str(Deaths_Arma_Measure["mape"]))
print("..............................................................")
print("MAE ARIMA : " + str(Deaths_Arima_Measure["mae"]))
print("RMSE ARIMA : " + str(Deaths_Arima_Measure["rmse"]))
print("MAPE ARIMA : " + str(Deaths_Arima_Measure["mape"]))
print("\n")


#%% visualize Holt Winters
fig,(ax0,ax1) = plt.subplots(2,figsize=(12,8))

ax0.plot(finalDf['Cases'],label='Cases')
ax0.plot(finalDf['Cases_hw_tes_mul-mul'],label='Cases_hw_tes_mul-mul')
ax0.plot(finalDf['Cases_hw_tes_add-add'],label='Cases_hw_tes_add-add')
ax0.plot(finalDf['Cases_predict_hw_tes_mul'],label='Cases_predict_hw_tes_mul')
ax0.plot(finalDf['Cases_predict_hw_tes_add'],label='Cases_predict_hw_tes_add')

ax1.plot(finalDf['Deaths'],label='Deaths')
ax1.plot(finalDf['Deaths_hw_tes_add'],label='Deaths_hw_tes_add')
ax1.plot(finalDf['Deaths_predict_hw_tes_add'],label='Deaths_predict_hw_tes_add')
ax1.plot(finalDf['Deaths_hw_tes_mul'],label='Deaths_hw_tes_mul')
ax1.plot(finalDf['Deaths_predict_hw_tes_mul'],label='Deaths_predict_hw_tes_mul')

ax0.legend()
ax1.legend()

#plt.show()
#%% visualize AR

fig,(ax0,ax1) = plt.subplots(2,figsize=(12,8))

ax0.plot(finalDf['Cases'],label='Cases')
ax0.plot(finalDf['Cases_predict_ar'],label='Cases_predict_ar')

ax1.plot(finalDf['Deaths'],label='Deaths')
ax1.plot(finalDf['Deaths_predict_ar'],label='Deaths_predict_ar')

ax0.legend()
ax1.legend()

#plt.show()
#%% visualize ARIMA

fig,(ax0,ax1) = plt.subplots(2,figsize=(12,8))

ax0.plot(finalDf['Cases'],label='Cases')
ax0.plot(finalDf['Cases_predict_arima'],label='Cases_predict_arima')

ax1.plot(finalDf['Deaths'],label='Deaths')
ax1.plot(finalDf['Deaths_predict_arima'],label='Deaths_predict_arima')

ax0.legend()
ax1.legend()

#plt.show()
#%% visualize ARMA

fig,(ax0,ax1) = plt.subplots(2,figsize=(12,8))

ax0.plot(finalDf['Cases'],label='Cases')
ax0.plot(finalDf['Cases_predict_arma'],label='Cases_predict_arma')

ax1.plot(finalDf['Deaths'],label='Deaths')
ax1.plot(finalDf['Deaths_predict_arma'],label='Deaths_predict_arma')

ax0.legend()
ax1.legend()

#plt.show()
#%% print screen (Prediction)
pd.set_option("display.max_rows", None, "display.max_columns", None)

cols = list(finalDf.columns)

for i in cols:
    print(" Results : "+i)
    print(finalDf[finalDf[i].notna()][i])

#%% save csv file

finalDf.to_csv("predict.csv")


#%% show plots finally ("designed for idle :)")
show()


#%%optimization tryna
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#import itertools
#
#d = range(0,1)
#p=q=range(0,15)
#pdq = list(itertools.product(p,d,q))
#train , test = df.iloc[:60,:] , df.iloc[60:,:]
#
#best_pred = list()
#for param in pdq:
#    try:
#        model_arima = SARIMAX(train.Deaths,order=param)
#        model_arima_fit = model_arima.fit()
#        best_pred.append([model_arima_fit.aic,param])
#        print(param,model_arima_fit.aic)
#    except:
#        continue
#    
#best_pred = np.array(best_pred)
#aicc = list()
#
#for i in range(len(best_pred)-1):
#  value = float(best_pred[i][0])
#  aicc.append(value)
#
#for i in range(len(best_pred)-1):
#  if best_pred[i][0] == np.array(aicc).min():
#    print("best model:",best_pred[i][1])
#    print("best model aic score:",np.array(aicc).min())
#    parameters = best_pred[i][1]
#    break
#  else:
#    continue
#
#def sarimax(data,p,d,q,testRate=0.2):
#    dataPos = data[data>0]
#    dataPosLen = len(dataPos)
#    dataPos = pd.to_numeric(dataPos,downcast='float')
#    
#    splitIndex = int(dataPosLen*(1-testRate))
#    train = dataPos[:splitIndex]
#    test = dataPos[splitIndex:]
#
#    model = SARIMAX(train,order=(p,d,q))
#    pred = model.predict(model.params_complete,start=totalIdx[dataLen-dataPosLen+splitIndex],end=totalIdx[-1])
#    pred = pd.DataFrame(pred)
#    maev = mae(test,pred[0][:len(test)])
#    rmsev = rmse(test,pred[0][:len(test)])
#    mapev = mape(pred[0][:len(test)],test)
#    
#    measure = {"mae":maev,"rmse":rmsev,"mape":mapev}
#    
#    return pred,measure
#
#Cases_sarimax,Cases_Arma_Measure = sarimax(data=df['Cases'],p=2,q=0,d=9)
#Deaths_sarimax,Deaths_Arma_Measure = sarimax(data=df['Deaths'],p=3,q=0,d=0)
#Cases_sarimax.rename(columns={0:"Cases_predict_sarimax"},inplace=True)
#Deaths_sarimax.rename(columns={0:"Deaths_predict_sarimax"},inplace=True)
#finalDf = pd.concat([finalDf,Cases_sarimax,Deaths_sarimax],axis=1)
#
#fig,(ax0,ax1) = plt.subplots(2,figsize=(12,8))
#
#ax0.plot(finalDf['Cases'],label='Cases')
#ax0.plot(finalDf['Cases_predict_sarimax'],label='Cases_predict_sarimax')
#
#ax1.plot(finalDf['Deaths'],label='Deaths')
#ax1.plot(finalDf['Deaths_predict_sarimax'],label='Deaths_predict_sarimax')
#
#ax0.legend()
#ax1.legend()
#
#plt.show()
#%%  get 1 and 5 in req list for total case ande total death values
## if doesnt big change on worldometer , it is going to work fine.

import re
import string
import pandas as pd 
import requests
import pandas as pd

class getDataFromWorldometer:
    def __init__(self,url):
        self.req = requests.get(url)
        self.req = self.cleanHtml(self.req.text)
    
    def cleanHtml(self,raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        lstClean = [ str(x).replace(";","").replace("(","").replace(")","").replace(",",", ") for x in cleantext.strip().split("Highcharts.chart")]
        return lstClean

    def exportData(self,arg):
        index = 0;
        for i in range(len(arg)):
            if arg[i] == " ":
                index = i 
                break
        
        data = []
        deneme = [x.strip().split(",") for x in str(arg[index:]).strip().replace("{","").replace("}","").split(":")]
        
        for a in range(len(deneme)):
            for b in range(len(deneme[a])):
                deneme[a][b] = str(deneme[a][b]).translate(str.maketrans("","",string.punctuation))
        
        for i in range(len(deneme)):
            if len(deneme[i])>10:
                data.append(deneme[i])
                
        data[0].pop()
        data[1].pop()
        data = {"date":data[0],"values":data[1]}
        return data
        
    def handleData(self):
        Cases = self.exportData(self.req[1])
        Deaths = self.exportData(self.req[5])
#        dfCases = pd.DataFrame(Cases[],
        dfC = pd.DataFrame(Cases["values"],index=Cases["date"],columns=["Cases"])
        dfD = pd.DataFrame(Deaths["values"],index=Deaths["date"],columns=["Deaths"])
        dfC = pd.concat([dfC,dfD],axis=1)
        return dfC

## paste your country page on worldometer
url = "https://www.worldometers.info/coronavirus/country/hungary/"
data = getDataFromWorldometer(url)
data = data.handleData()
    


##!!! data frame index kolonu date time a çevir ve frekansını günlük olarak ayarla !!!


#df = pd.DataFrame(data["Deaths"],index=data["date"])
#df.index = pd.to_datetime(df.index,format="%m %d")
#df.index.freq = 'D'
