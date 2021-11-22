import  jieba
from    gensim.models   import  word2vec

def tokensize(path):
    with open(path,encoding='utf-8') as f1:
        document = f1.read()
        document_cut = jieba.cut(document)
        result=' '.join(document_cut)
    return result

def modelTrain(train):
    sentence=word2vec.LineSentence(train)
    model=word2vec.Word2Vec(sentences=sentence,vector_size=100,window=20,min_count=1)
    #参数设置说明：
    model.save(r".\word2vec.model")
    model.wv.save_word2vec_format(r".\w2v.txt")
def write(result):
    with    open(r".\sentence.txt",'w',encoding='utf-8') as f2:
        f2.write(result)
if __name__=="__main__":
    path=r".\data.txt"
    result=tokensize(path)
    write(result)
    train=r".\sentence.txt"
    modelTrain(train)