
# coding: utf-8

# In[ ]:
import sys
sys.path.append('/home/wq/yqzwqa')

from transformers import pipeline, AutoModelForQuestionAnswering, BertTokenizer, AutoConfig
#QuestionAnsweringPipeline
from question_answering_pipeline import QuestionAnsweringPipeline
import torch
import re
import pandas as pd
import pickle
#from apex import amp
from rouge.rouge import Rouge
from difflib import SequenceMatcher
import copy
import thulac
import json

test_N = pd.read_csv("./test_0512.csv") 
print(test_N.shape)
#test_N = test_N.iloc[:10,:]
"""
model = AutoModelForQuestionAnswering.from_pretrained('./torch0512/checkpoint-4/')
tokenizer = BertTokenizer.from_pretrained('./torch0512/checkpoint-4/')
#model, optimizer = amp.initialize(model.to(device='cuda'), None)

Qa_pipe = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)



test_preds = Qa_pipe(
question=test_N.question.tolist(),
context=test_N['text'].str.replace(r'\s+', ' ').str.strip().str.slice(stop=5000).tolist(), 
max_answer_len=400, max_seq_len=512, max_question_len=80, doc_stride=128, topk=20)
#print(test_preds[0])
print(len(test_preds))
#print(test_preds)
#这里需要处理下结果
"""
#test_preds = [test_preds[i*20:(i+1)*20] for i in range(len(test_preds)//20)]
#pickle.dump(test_preds, open('test_N_0514.p', 'wb'))
test_preds = pickle.load(open("test_N_0514.p","rb"))

thu = thulac.thulac(seg_only=True, user_dict='user_dict.txt')

import sys
sys.setrecursionlimit(10000)

rouge = Rouge()

def spacify(s):
    return re.sub(r'\s+', ' ', ' '.join(list(s)))

def get_result(model_preds, df_search, a, b, c, w1, w2, w3, w4):
    model_preds = copy.deepcopy(model_preds)
    
    max_answer_len = 2500
    answers = []
    
    for i, p in enumerate(model_preds):
        cands = []
        #print(p)
        
        min_null_score = 10000
        for j, d in enumerate(p):
            if d['answer'] == '':
                min_null_score = min(min_null_score, d['score'])
            else:
                d['score1'] = c * d['score'] + (1 - c) * (1 - d['null_score'])
            
                
#                 if len(d['answer']) < 2 or len(d['answer']) > max_answer_len:
#                     d['score1'] = max(0, d['score1'] - 0.1)
                cands.append(d)

        cands = sorted(cands, key=lambda x:x['score1'], reverse=True)
        
        s1 = cands[0]['score1'] if len(cands) > 0 else 0
        s2 = cands[1]['score1'] if len(cands) > 1 else 0
        s3 = cands[2]['score1'] if len(cands) > 2 else 0
        s4 = cands[3]['score1'] if len(cands) > 3 else 0
        
        score_context = w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4

        best_answer = cands[0]['answer'].strip()

        if min_null_score > 1:
            min_null_score = 0
            
        if len(best_answer) < 3 or len(best_answer) > max_answer_len:
            score_context = score_context - 0.5
            
        answers.append((best_answer, min_null_score, score_context))
        
    df_preds = pd.DataFrame(answers, columns=['pred', 'null_prob', 'prob'])

    df_search_pred = pd.concat([df_search.reset_index(drop=True), df_preds], axis=1)
#         df_search_pred = df_search_pred[df_preds['null_prob'] < thres]

    df_search_pred['score_all'] = df_search_pred['bm25_score'] * (1 - a - b) / 100 + df_search_pred['prob'] * a + (1 - df_search_pred['null_prob']) * b

    df_pred_select = []
    for k, _df in df_search_pred.groupby('id'):
        df_pred_select.append(_df.sort_values('score_all', ascending=False).iloc[0])

    df_pred_select = pd.DataFrame(df_pred_select)

    answer_pred_refine = []
    for row in df_pred_select.itertuples():
        matches = [m for m in SequenceMatcher(a=row.text, b=row.pred).get_matching_blocks() if m.size > 0]
        refine = row.text[matches[0].a: matches[-1].a + matches[-1].size]
        
        old_refine = copy.copy(refine)
        
        refine = refine.strip('，').strip(':').strip('：').strip('（').strip('；').strip('、').strip()
        
        if re.sub(r'\s+', '', refine) != re.sub(r'\s+', '', row.pred):
            refine = row.pred
        
        if len(refine) > 5 and len(re.findall('[。，：]', refine[:4])) > 0:
            for f in re.finditer('[。，：、]', refine[:4]):
                refine = refine[f.span()[0] + 1:]
                break
                
            print('----截取开头标点----')
        
        
        if re.match(r'^[的].*', refine):
            refine = refine[1:]
            print('----去掉开头停用词----')
            
        if matches[0].a != 0:
            tmp = row.text[matches[0].a - 1: matches[-1].a + matches[-1].size]
            tw = [w[0] for w in thu.cut(tmp)]
            rw = [w[0] for w in thu.cut(refine)]
            
            if len(rw) > 0 and rw[0] != tw[0] and rw[0] in tw[0]: # 少了一个字
                print('----去掉开头不完整词语----')
                refine = tmp[len(tw[0]):]
                
#         if matches[-1].a + matches[-1].size < len(row.text):
#             tmp = row.text[matches[0].a: matches[-1].a + matches[-1].size + 1]
#             tw = [w[0] for w in thu.fast_cut(tmp)]
#             rw = [w[0] for w in thu.fast_cut(refine)]
#             if len(rw) > 0 and rw[-1] != tw[-1] and rw[-1] in tw[-1]:
#                 print('----去掉结尾不完整词语----')
#                 refine = tmp[: len(tmp) - len(tw[-1])]
                
        refine = refine.strip('，').strip(':').strip('：').strip('（').strip('；').strip('、').strip()
                
        if old_refine != refine:
            print('old:', old_refine)
            print('new:', refine)
            
        answer_pred_refine.append(refine)

    df_pred_select['answer_pred_refine'] = answer_pred_refine

    return df_pred_select


df_pred_select = get_result(test_preds[:20], test_N.iloc[:20,:], 0.3, 0, 0, 1, 0, 0, 0)
print(df_pred_select)
"""
df_rst = df_pred_select[['id', 'docid', 'answer_pred_refine']]
df_rst.columns = ['id', 'docid', 'answer']
df_rst.to_csv('./0515.csv', sep='\t', index=None)

print("********************test is finished**********************")
"""
