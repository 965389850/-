import re
import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import wordcloud as wc
import pandas as pd

def clearTxt(line:str):
    '''清洗文本'''
    if(line != ''):
        line = line.strip()
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
        return line
    print("文本为空！")
    return None

def sent2word(line):
    '''文本切割'''
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


def Cloud_words(words, path):
    '''# 生成词云'''
    # 引入字体
    # mask = np.array(Image.open('love.png'))
    # image_colors = ImageColorGenerator(mask)
    #从文本中生成词云图
    cloud = wc.WordCloud(
                          font_path="/System/Library/fonts/PingFang.ttc",#设置字体 
                          background_color='white', # 背景色为白色
                          height=600, # 高度设置
                          width=900, # 宽度设置
                          scale=20, # 长宽拉伸程度程度设置为20
                          prefer_horizontal=0.0, # 调整水平显示倾向程度
                          max_font_size=100, #字体最大值 
                          max_words=1000, # 设置最大显示字数为2000
                          relative_scaling=0.3, # 设置字体大小与词频的关联程度为0.3
                         )
    # 绘制词云图
    mywc = cloud.generate(words)
    plt.imshow(mywc)
    plt.axis('off')
    mywc.to_file(path)


def clean_and_plot(data, pic_out_path,print_weight):
    '''
    输入文本信息，以及云图输出路径；输出词频前10的词和处理后的用于聚类的数据
    '''
    clean_data = [item for item in data]
    # 清洗文本
    clean_data = [clearTxt(item) for item in clean_data]
    #文本切割
    clean_data = [sent2word(item) for item in clean_data]
    Cloud_words(','.join(clean_data), pic_out_path)
    vectorizer = CountVectorizer()
    a = vectorizer.fit_transform(clean_data)
    x = vectorizer.get_feature_names_out()
    print('输出词袋内容：\n',x)
    word = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:10]
    print('输出词频前十的词：\n',word)

    tf_idf_transformer = TfidfTransformer()
    tfidf = tf_idf_transformer.fit_transform(a)
    print(tfidf)
    tfidf_matrix = tfidf.toarray()

    if print_weight == True:
        word = vectorizer.get_feature_names_out()
        for i in range(len(tfidf_matrix)):
            for j in range(len(word)):
                print(word[j],tfidf_matrix[i][j])
    
    return word,tfidf_matrix


if __name__ == "__main__":
    data = pd.read_csv('%23俄乌战争%23.csv')
    print(data.shape)
    # 清除重复数据
    data.drop_duplicates('用户昵称',keep='first',inplace=True)

    clean_and_plot(data, 'test.png', True)