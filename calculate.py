import gensim
from sklearn.decomposition import PCA
import sns
import numpy as np
def choose(model):
    result = int(input("①如果计算两个词语的相关性请输入1\n"
                       "②如果计算与一个词语最相关的词语请输入2\n"
                       "③如果计算与指定词类比最相似的5个词请输入3\n"
                       "④如果4，表示此刻将会对向量降维\n"))
    if result == 1:
        word1   = input("请输入第一个词语:\n")
        word2   = input("请输入第二个词语:\n")
        similarity(model, word1, word2)
    elif result == 2:
        word = input("请输入中心词:\n")
        num =   int(input("请输入想要查找的最相关词的数量:\n"))
        connectivity(model,word,num)
    elif result == 3:
        category1 = input("请输入理想类别的第一个词:\n")
        category2 = input("请输入理想类别的第二个词:\n")
        word = input("请输入指定词:\n")
        num = input("请输入你想寻找的词的个数:\n")
        category(model, category1, category2, word, num)
    else :
        city = ['江苏','南京','成都','四川','郑州','河南','郑州','甘肃','兰州']
        vec = []
        for index in range(len(city)):
            vec.append(model.wv[city[index]])
        vec = np.asarray(vec)
        pca = PCA(n_components=2)
        result = pca.fit_transform(vec)
        sns.scatterplot(x=result[:, 0], y=result[:, 1])
        #print(result)
def similarity(wv_model,word1,word2):
    r = wv_model.wv.similarity(word1, word2)
    print(word1+'和'+word2+"的相关性是："+str(r))
def connectivity(wv_model,word,num):
    r = wv_model.wv.most_similar(positive=[word],topn=num)
    for i in range(num):
        print("Top", str(i+1), ":", str(r[i][0]),"相关率 ：", '\t', str(r[i][1]))
def category(model,category1,category2,word,num):
    r = model.wv.most_similar(positive=[category1, category2], negative=[word], topn=num)
    for i in range(num):
        print("Top", str(i+1), ":", str(r[i][0]),"相关率 ：", '\t',str(r[i][1]))


def loadmodel():
    print("模型加载中...")
    path = r".\word2vec.model"
    wv_model = gensim.models.Word2Vec.load(path)
    print("模型加载完毕...")
    return wv_model
#def loadvector():
    #print("词向量加载中...")
    #path = r".\w2v.txt"
    #name=[]
    #index=[]
    #words = ['中国','法国','日本','美国','共产党','人民','邓小平','香港','社会','冲突','改革','和平']
    #with open(path,'r',encoding='utf-8') as f:
     #   doc=f.readlines()
      #  for m in doc:
       #     m=m.split(' ')
        #    name.append(m[1])
        #for i in words:
         #   index.append.m.find(i)
        #for i in len(words):
         #   vector=

    #print("词向量加载完毕...")
    #return vector

if __name__=="__main__":
    model = loadmodel()
    #vector = loadvector()
    choose(model)
    #pca(vector)


