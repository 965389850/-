import re
import pandas as pd
import jieba
from snownlp import SnowNLP

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 情感分类
def fenlei(request):

    j = '#印尼官方回应中国游客巴厘岛遇害事件#】5月1日，位于印尼巴厘岛的洲际酒店发生一起惊悚的命案，两名中国游客不幸身亡。印尼旅游和创意经济部长在接受CGTN记者采访时表示，对中国游客遇害事件深感悲痛、深表哀悼，印尼警方正在全力调查该事件，并已形成初步报告。他还表示，#巴厘岛#警方与执法部门已经组成特别小组，将尽快调查到底、查明真相，坚决杜绝此类事件的再次发生。LCGTN记者团的微博'
    s = SnowNLP(j)
    print(s.sentiments)

    for item in tqdm(WeiBo.objects.all()): # 按行读取数据
        emotion = '正向' if SnowNLP(item.content).sentiments >0.45 else '负向'
        WeiBo.objects.filter(id=item.id).update(emotion=emotion)
    # return JsonResponse({'status':1,'msg':'操作成功'} )

# 清洗文本
def clearTxt(line:str):
    if(line != ''):
        line = line.strip()
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
        return line
    print("文本为空！")
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
    #print(word_cut)
    segSentence = ''
    for word in word_cut:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()

if __name__== "__main__" :
    #读取数据
    data = pd.read_csv('%23新冠肺炎%23.csv')
    print(data['用户昵称'].head(5))



    # 清洗文本
    clean_data = [item for item in data['用户昵称']]
    clean_data = [clearTxt(item) for item in clean_data]
    clean_data = [sent2word(item) for item in clean_data]

    # print(clean_data)

    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tfidf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(clean_data))
    # 获取词袋模型中的所有词语
    tfidf_matrix = tfidf.toarray()
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    x=vectorizer.fit_transform(clean_data)
    #print(tfidf.toarray()[len(tfidf)/2][len(tfidf)/2])
