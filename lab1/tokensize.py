import jieba
sentence=jieba.cut("庞远心今天取了7个快递",cut_all=False)
print("/".join(sentence))
