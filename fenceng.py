import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import functools
import numpy as np
from snownlp import SnowNLP
from snownlp import sentiment
import csv

# 清洗文本
def clearTxt(line:str):
    if(line != ''):
        line = line.strip()
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
        return line
    return None

#文本切割
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)
    # 去停用词
    stopwords1 = [line.strip() for line in open("stopwords/baidu_stopwords.txt", 'r', encoding="utf-8").readlines()]
    stopwords2 = [line.strip() for line in open("stopwords/cn_stopwords.txt", 'r', encoding="utf-8").readlines()]
    stopwords3 = [line.strip() for line in open("stopwords/hit_stopwords.txt", 'r', encoding="utf-8").readlines()]
    stopwords4 = [line.strip() for line in open("stopwords/scu_stopwords.txt", 'r', encoding="utf-8").readlines()]
    stopwords1.extend(stopwords2)
    stopwords1.extend(stopwords3)
    stopwords1.extend(stopwords4)
    # print(stopwords1)
    word_cut = [i for i in segList if i not in stopwords1 and len(i)!=1]
    # print(word_cut)
    segSentence = ''
    for word in word_cut:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()

#数据清洗
def cleandata(data):
    # 清洗文本
    clean_data = [item for item in data]
    clean_data = [clearTxt(item) for item in clean_data]
    clean_data = [sent2word(item) for item in clean_data]
    return clean_data

def emotion(words):
# 评论情感分析
    # items=words.astype(str).tolist()
    # print(items)
 
    D=[]
    for i in range(len(words)):
        print(words[i])
        s=SnowNLP(words[i])
        t=s.sentiments
        print(t)
        a=[words[i],t]
        D.append(a)
    print(D)
    with open('emotion.csv','a',encoding='utf-8-sig',newline='')as f1:
        write=csv.writer(f1)
        write.writerows(D)
    # print("s = ", s)
    # print("s类型", type(s))
    # print("1、中文分词:\n",s.words)   
    # print("2、词性标注:\n",s.tags)
    # print("3、情感倾向分数:\n",s.sentiments)
    # print("4、转换拼音:\n",s.pinyin)
    # print("5、输出前4个关键词:\n",s.keywords(4))
    # print("6、输出关键（中心）句:\n",s.summary(1))
    # print("7.1、输出tf:\n",s.tf)
    # print("7.2、输出idf:\n",s.idf)
    # print("8、繁简体转换:\n",s.han)
    return None

#计算tf-idf权值
def compute(clean_data):
    # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tfidf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(clean_data))
    print( "词频矩阵为：\n",tfidf.toarray())
     # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names_out()
    x=vectorizer.fit_transform(clean_data)
    # print('输出词袋内容：\n',x)
    word = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:10]
    # print('输出词频前十的词：\n',word)
    return tfidf


#文本词语转词频矩阵，文本数据做聚类
def  kmeansPlot(tfidf, num):
    tfidf_array = tfidf.toarray()
    # kmeans聚类
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=num, n_init='auto')
    result = clf.fit(tfidf_array)

    #预测聚类结果
    y_pred = clf.predict(tfidf_array)
    # tfidf_array = np.float64(tfidf_array)
    result_list  = list(y_pred)
    print('预测聚类结果名单为：\n',result_list)

    #中心点
    # print(len(clf.cluster_centers_))#对于矩阵来说len是行
    print("输出中心点：\n",clf.cluster_centers_)#每一类的中心点
 
    #聚类评估指标，距离越小说明簇分的越好
    print("计算簇中某一点到簇中距离的和: \n",clf.inertia_)
    print("每个点所属簇标签: \n",clf.labels_)
    return None

# 聚类结果可视化
def view(num, tfidf):
    tfidf_array = tfidf.toarray()
    colors_list = ['teal', 'skyblue', 'tomato', 'black']
    labels_list = ['0', '1', '2', '3']
    markers_list = ['o', '*', 'D', '1']  # 分别为圆、星型、菱形
    #画中心点
    # for i in range(num):
    #     plt.scatter(clf.cluster_centers_[i], clf.cluster_centers_[i], s=300, c=colors_list[i],
    #         label=labels_list[i], marker=markers_list[i])
    # plt.show()
    for i in range(num):
         plt.scatter(tfidf_array[i], tfidf_array[i], c=colors_list[i],
            label=labels_list[i], marker=markers_list[i])
    plt.show()
    return  None

if __name__ == "__main__":
    data=pd.read_csv("%23俄乌战争%23.csv")
    num = 4
    # words = cleandata(data['用户昵称'])
    # list = list(data['用户昵称'])
    emotion(data['用户昵称'])
    #计算tf-idf权值
    # tfidf = compute(cleandata(data['用户昵称']))
    # kmeansPlot(tfidf, num)
    # view(num, tfidf)
