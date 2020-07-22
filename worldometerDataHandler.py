# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:51:52 2020

@author: atidem
"""

##  get 1 and 5 in req list for total case ande total death values(for U.S. 2 and 6),
##  replaceable for different data    "req[1]" 
##  if doesnt big change on worldometer , it is going to work fine.

import re
import string
import pandas as pd 
import requests

class GetDataFromWorldometer:
    def __init__(self,url):
        ## actually get .html file from url
        self.url = url
        self.req = requests.get(self.url)
        self.req = self.cleanHtml(self.req.text)
         
    
    def cleanHtml(self,raw_html):
        ## clean html tags and split
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        lstClean = [ str(x).replace(";","").replace("(","").replace(")","").replace(",",", ") for x in cleantext.strip().split("Highcharts.chart")]
        return lstClean

    def exportData(self,arg):
        ## export total cases and total deaths data
        index = 0;
        for i in range(len(arg)):
            if arg[i] == " ":
                index = i 
                break
        
        data = []
        deneme = [x.strip().split(",") for x in str(arg[index:]).strip().replace("{","").replace("}","").split(":")]
        
        for a in range(len(deneme)):
            for b in range(len(deneme[a])):
                deneme[a][b] = str(deneme[a][b]).translate(str.maketrans("","",string.punctuation)).strip()
        
        for i in range(len(deneme)):
            if len(deneme[i])>10:
                data.append(deneme[i])
        
        data[0].pop()
        data[1].pop()
        data[1] = [float(x) for x in data[1]]
        data = {"date":data[0],"values":data[1]}
        return data
        
    def handleData(self):
        ## return dataframe from modified data 
        if self.url.split("/")[-2] == "us":
            Cases = self.exportData(self.req[2])
            Deaths = self.exportData(self.req[6])
        elif self.url.split("/")[-2] == "canada":
            Cases = self.exportData(self.req[3])
            Deaths = self.exportData(self.req[7])
        else:
            Cases = self.exportData(self.req[1])
            Deaths = self.exportData(self.req[5])
            
        dfC = pd.DataFrame(Cases["values"],index=Cases["date"],columns=["Cases"])
        dfD = pd.DataFrame(Deaths["values"],index=Deaths["date"],columns=["Deaths"])
        dfC = pd.concat([dfC,dfD],axis=1)
        dfC.index = pd.date_range(pd.to_datetime(str(dfC.index[0]+ " 2020"),format="%b %d %Y"),periods=len(dfC),freq='D')
        dfC.index.freq = 'D'
        return dfC

# çekilen veriler kontrol edilip handle data kısmında req[x] kısmı değiştirilerek uyum sağlanabilir.
# total cases ve total deaths değişkenleri incelenerek doğru X değeri seçilebilir.
# doğru X değeri için altta bulunan kodda aa değişkeni incelenip doğru dataların çağırılması sağlanabilir.

#url = "https://www.worldometers.info/coronavirus/country/canada/"
#getData = GetDataFromWorldometer(url)
#aa = getData.req
#df = getData.handleData()
