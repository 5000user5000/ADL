import json
def tranfer2Json(path):
    times = 1
    gloveJson = {}
    with open(path,'r',encoding='UTF-8') as f:
        line = f.readline()
        while line is not None and line != '':
            print(times)
            label = line[0]
            vec = line[1:].split()
            gloveJson[label] = vec
            line = f.readline()
            times+=1
        with open('glove.json','w') as f2:
            json.dump(gloveJson,f2,indent=4)
    


tranfer2Json('glove.840B.300d.txt')
