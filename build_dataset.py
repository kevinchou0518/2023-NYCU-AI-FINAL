import requests
import json 
from bs4 import BeautifulSoup  

urls = []
count = 0
for month in range(1,13):
    for day in range(1,28):
        u = 'https://www.ettoday.net/news/news-list-2023-'+str(month)+'-'+str(day)+'-'+str(0)+'.htm'
        res = requests.get(u)
        soup = BeautifulSoup(res.content, "lxml")
        soup = soup.find("div", class_="part_list_2")
        domian = "https://www.ettoday.net"
        print(u)
        
        for a in soup.find_all("h3"):
            urls.append(domian+a.a['href'])
            count+=1
        if(count > 23000):
            break
allcontent = []
for u in urls:
    content = []
    res = requests.get(u)
    
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
        #print(content)
        allcontent.append(content)
    except:
        pass
print(len(allcontent))
js_train = []
js_test = []
count = 0
with open('train.txt', 'w', newline = '') as train, open('test.txt', 'w', newline = '') as test:
    for c in allcontent:
        if(len(c) > 3):
            count+=1
            if count % 10 == 0:
                temp = ""
                for sentence in c:
                    temp+=sentence
                js_test.append({"article" : temp})
            else:
                temp = ""
                for sentence in c:
                    temp+=sentence
                js_train.append({"article" : temp})
        if(count == 22000):
            break
    json.dump(js_train, train)
    json.dump(js_test, test)
