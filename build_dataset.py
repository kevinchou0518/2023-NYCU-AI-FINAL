import requests
import json 
from tqdm import tqdm
from bs4 import BeautifulSoup  

urls = []
title = []
print("Accessing all news url")
for year in tqdm(range(2020,2023)):
    for month in tqdm(range(1,13)):
        for day in range(2,28):
            if(day%2 == 0):
                u = 'https://www.ettoday.net/news/news-list-'+str(year)+'-'+str(month)+'-'+str(day)+'-'+str(0)+'.htm'
                res = requests.get(u)
                soup = BeautifulSoup(res.content, "lxml")
                soup = soup.find("div", class_="part_list_2")
                domian = "https://www.ettoday.net"
                for a in soup.find_all("h3"):
                    t = a.a.string
                    if t != None:
                        #print(domian+a.a['href'])
                        urls.append(domian+a.a['href'])
                        title.append(t)
urllist = []
allcontent = []
titlelist = []
print("Access all news text")
for u in tqdm(range(len(urls))):
    content = []
    res = requests.get(urls[u])
    soup = BeautifulSoup(res.content, "lxml")
    try:
        soup = soup.find("div", class_="story")
        for a in soup.find_all("p"):
            p = a.string
            if p != None:
                p = p.split('／')
                if len(p) > 1:
                    content.append(p[1])
                else:
                    content.append(p[0])
        allcontent.append(content)
        urllist.append(urls[u])
        titlelist.append(title[u])
    except:
        pass
index = 0
js_test = []
js_eval = []
js_train = []

with open('train.json', 'w', encoding='UTF-8') as train, open('eval.json', 'w', encoding='UTF-8') as eval, open('test.json', 'w', encoding='UTF-8') as test:
    print("arrange the data and dump into json file")
    for c in tqdm(range(len(allcontent))):
        if(len(allcontent[c]) > 3):
            index+=1
            if index % 10 == 0:
                temp = ""
                for sentence in allcontent[c]:
                    temp+=sentence
                js_test.append({"id" : index, "title" : titlelist[c], "url" : urllist[c], "article" : temp})
            elif index % 10 == 9:
                temp = ""
                for sentence in allcontent[c]:
                    temp+=sentence
                js_eval.append({"id" : index, "title" : titlelist[c], "url" : urllist[c], "article" : temp})
            else:
                temp = ""
                for sentence in allcontent[c]:
                    temp+=sentence
                js_train.append({"id" : index, "title" : titlelist[c], "url" : urllist[c], "article" : temp})
    print("dump to train.json")
    for data in js_train:
        train.write(json.dumps(data , ensure_ascii=False) + "\n")
    print("dump to test.json")
    for data in js_test:
        test.write(json.dumps(data , ensure_ascii=False) + "\n")
    print("dump to eval.json")
    for data in js_eval:
        eval.write(json.dumps(data , ensure_ascii=False) + "\n")
print("finish")

