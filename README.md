
比赛链接：https://www.datafountain.cn/competitions/424

采用框架：修改transformers框架里面的run_squad.py文件代码，链接 https://github.com/huggingface/transformers/tree/master/examples 直接按要求安装transformers即可。

疫情问答助手baseline思路比较简单：把数据整成官方训练数据json格式训练即可。预测文件难点在于找到答案文档。这里基于bm25检索文档，为了提高答案召回率可以按照bm25得分多取几个，这里比较佛系，我取了个top3预测。最终得分提交评价指标得分0.58+。

这题难点在于不用外部数据情况下,针对预测数据既能准确找到答案所在文档又能使文档尽可能的短。第一次接触问答欢迎有兴趣的同学多交流。



