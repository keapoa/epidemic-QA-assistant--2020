
# coding: utf-8

# In[1]:


import pandas as pd
import json
import jieba.posseg as pseg
from gensim import corpora
from gensim.summarization import bm25

#数据读取

train = pd.read_csv(open(r"C:\Users\qian_wang\Desktop\疫情\NCPPolicies_train_20200301.csv",encoding = "utf-8"),delimiter = "\t")
test = pd.read_csv(open(r"C:\Users\qian_wang\Desktop\疫情\NCPPolicies_test.csv",encoding = "utf-8"),delimiter = "\t")
doc = pd.read_csv(open(r"C:\Users\qian_wang\Desktop\疫情\NCPPolicies_context_20200301.csv",encoding = "utf-8"),delimiter = "\n")
doc_pro = pd.DataFrame()
doc_pro["docid"] = [text[0].split("\t")[0] for text in doc.values]
doc_pro["text"] = [text[0].split("\t")[1] for text in doc.values]
#doc_pro去重
#doc_pro.drop_duplicates(subset = ["text"],keep='first',inplace = True)
train_doc = pd.merge(train,doc_pro,how = "left",on = "docid")

#生成训练以及验证数据文件
def data_json(data):
    res = {}
    count=0
    data_dic = []
    for i in range(len(data)):
        
        question = " ".join(data.iloc[i,2])
        text = " ".join("".join(data.iloc[i,4].split()))
        answers = " ".join("".join(data.iloc[i,3].split()))
        answers_start = text.find(answers)
        if answers_start==-1:
            count+=1
            print(i)
        dic = {}
        dic["context"] = text
        
        dic["qas"] = [{"answers":[{"answer_start":answers_start,"text":answers}],"question":question,"id":data.iloc[i,0]}]
        data_dic.append({"title":"疫情","paragraphs":[dic]})
                      
    res["data"] = data_dic
    print("么有答案的样本{}".format(count))
    return res
train_json_08 = data_json(train_doc.iloc[:4000,:])

with open(r'C:\Users\qian_wang\Desktop\疫情\train_0.8.json','w',encoding = 'utf-8') as f:
    json.dump(train_json_08,f,ensure_ascii=False)
f.close()
valid_json_02 = data_json(train_doc.iloc[4000:5000,:])
with open(r'C:\Users\qian_wang\Desktop\疫情\valid_0.2.json','w',encoding = 'utf-8') as f:
    json.dump(valid_json_02,f,ensure_ascii=False)
f.close()


# In[2]:


#bm25训练

stop_words = []

for i in open(r"C:\Users\qian_wang\Desktop\疫情\stop_words.txt",encoding = "utf-8").readlines():
    stop_words.append(i.strip())
stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
def tokenization(texts):
    result = []
    for word in texts["text"].values:
        words = pseg.cut(word)
        out = []
        for word, flag in words:
            
            if flag not in stop_flag and word not in stop_words:
            #if flag not in stop_flag:
                out.append(word)
        result.append(out)
    return result
#构建语料库
corpus = []
corpus = tokenization(doc_pro)
#构建bm25模型
bm25Model = bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]),bm25Model.idf.keys())) / len(bm25Model.idf.keys())
#判断相关性
def top_correlation_index(query,bm25Model,average_idf,stop_flag):

    words = pseg.cut(query)
    out = []
    for word, flag in words:
        if flag not in stop_flag:
                out.append(word)
            
    scores = bm25Model.get_scores(out,average_idf)
    
    return scores.index(max(scores))

#构建docid-index

docid_index = {}
for i,docid in enumerate(doc_pro["docid"].values):
    docid_index[i] = docid


# In[5]:


#测试样本生成，这里取bm25top3样本连接预测
def topN_doc(query,bm25Model,average_idf,stop_flag,N):
    words = pseg.cut(query)
    out = []
    for word, flag in words:
        if flag not in stop_flag:
                out.append(word)
            
    scores = bm25Model.get_scores(out,average_idf)
    res = {}
    for i,v in enumerate(scores):
        res[i] = v
        
    return sorted(res.items(),key = lambda x:x[1],reverse = True)[:N]
def test_sample_docidN(test,docid_index,N):
    id_ = []
    question_ = []
    docid_ = []
    scores = []
    for i in range(test.shape[0]):
        dic = topN_doc(test["question"][i],bm25Model,average_idf,stop_flag,N)
        docids = []
        score_dic = []
        
        #添加分数，分数归一化
        sum_ = sum([i[1] for i in dic]) 
        for k,v in dic:
            docids.append(docid_index[k])
            score_dic.append(v/sum_) 
        id_.extend([test["id"][i]]*N)
        docid_.extend(docids)
        question_.extend([test["question"][i]]*N)
        scores.extend(score_dic)
    return pd.DataFrame({"id":id_,"question":question_,"docid":docid_,"bm25_score":scores})
def test_N(test,N):
    ids = []
    docids = []
    questions = []
    texts = []
    for i in range(0,test.shape[0],20):
        id_ = test["id"][i]
        docid_ = test["docid"][i]
        question_ = test["question"][i]
        text_ = ""
        text_temp = set(test["text"][i:i+N])
        for text in text_temp:
            text_+=text
        ids.append(id_)
        docids.append(docid_)
        questions.append(question_)
        texts.append(text_)
    return pd.DataFrame({"id":ids,"docid":docids,"question":questions,"text":texts})

test_sample_N =  test_sample_docidN(test,docid_index,20)
test_sample_N_docid = pd.merge(test_sample_N,doc_pro,how = "left",on = "docid")

test_final = test_N(test_sample_N_docid,3)
test_final["answer"] = ["1"]*test_final.shape[0]
test_doc = test_final[["id","docid","question","answer","text"]]
test_json = data_json(test_doc)
with open(r'C:\Users\qian_wang\Desktop\疫情\test_torch_0413_N3_json.json','w',encoding = 'utf-8') as f:
    json.dump(test_json,f,ensure_ascii=False)
f.close()


# In[6]:


#结果集生成提交文件
test_ = json.load(open(r"C:\Users\qian_wang\Desktop\疫情\predictions_0413.json",encoding = "utf-8"))
test_dic = {}
for i in test_.keys():
    test_dic[i] = "".join(test_[i].split())    
test_doc["answer"] = test_dic.values()
test_doc[["id","docid","answer"]].to_csv(r"C:\Users\qian_wang\Desktop\疫情\test_predictions_0413.csv",encoding = "utf-8",sep = "\t",index = None)

