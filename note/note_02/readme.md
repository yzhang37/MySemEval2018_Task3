# MySemEval2018_Task3 Experimental Logs #002 

接着 note_01 继续往后记录

## 2018-01-02

### 01. `liblinear` 整体 hc 运行结果

#### hc 总结

| 名称                      | 频次   |
| ----------------------- | ---- |
| nltk_unigram_t3_with_rf | 100  |
| wv_GloVe                | 100  |
| sentilexi               | 100  |
| ners_existed            | 90   |
| hashtag_t2              | 75   |
| wv_google               | 61   |
| hashtag_t5              | 51   |
| elongated               | 50   |
| nltk_bigram_t2          | 50   |
| hashtag_t3_with_rf      | 49   |
| emoticon                | 48   |
| nltk_bigram_t2_with_rf  | 31   |
| punction                | 26   |
| nltk_trigram_t5         | 24   |
| nltk_trigram_t3_with_rf | 12   |
| nltk_unigram_t2         | 10   |
| nltk_bigram_t3_with_rf  | 10   |


对所有的hc结果统计前十行，计算出来的结果为：

| 名称             | 选项                        | 频次    |
| -------------- | ------------------------- | ----- |
| `nltk_unigram` | `nltk_unigram_t3_with_rf` | `100` |
| `nltk_bigram`  | `nltk_bigram_t2`          | `50`  |
| `nltk_trigram` | `nltk_trigram_t5`         | `24`  |
| `hash_tag`     | `hashtag_t2`              | `75`  |
| `wv_GloVe`     | `wv_GloVe`                | `100` |
| `wv_google`    | `wv_google`               | `61`  |
| `sentilexi`    | `sentilexi`               | `100` |
| `ners_existed` | `ners_existed`            | `90`  |
| `elongated`    | `elongated`               | `50`  |
| `emoticon`     | `emoticon`                | `48`  |
| `punction`     | `punction`                | `26`  |

分别使用这些特征进行训练，得到的结果如下

#### 二分类：
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 61.14%     | 61.10%     | 61.07%     |
| Fold 2   | 65.05%     | 65.01%     | 64.98%     |
| Fold 3   | 62.74%     | 62.65%     | 62.60%     |
| Fold 4   | 67.27%     | 67.17%     | 67.13%     |
| Fold 5   | 69.32%     | 69.18%     | 69.13%     |
| Fold 6   | 68.41%     | 68.41%     | 68.41%     |
| Fold 7   | 62.51%     | 62.50%     | 62.50%     |
| Fold 8   | 63.20%     | 63.18%     | 63.17%     |
| Fold 9   | 62.93%     | 62.93%     | 62.92%     |
| Fold 10  | 64.69%     | 64.68%     | 64.67%     |
| **Mean** | **64.73%** | **64.68%** | **64.66%** |

#### 四分类:
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 50.35%     | 35.60%     | 36.11%     |
| Fold 2   | 37.85%     | 36.46%     | 36.26%     |
| Fold 3   | 45.49%     | 41.81%     | 42.18%     |
| Fold 4   | 33.77%     | 34.40%     | 33.76%     |
| Fold 5   | 36.51%     | 35.70%     | 35.52%     |
| Fold 6   | 47.89%     | 43.67%     | 43.99%     |
| Fold 7   | 45.20%     | 40.25%     | 41.32%     |
| Fold 8   | 44.77%     | 41.22%     | 42.16%     |
| Fold 9   | 45.23%     | 38.20%     | 38.99%     |
| Fold 10  | 38.37%     | 36.69%     | 36.78%     |
| **Mean** | **42.54%** | **38.40%** | **38.71%** |


## 2018-01-03

分别对 `AdaBoost`, `DecisionTree`, `NaiveBayes` 进行测试集训练

### 01. `AdaBoost`训练

| 特征               | 结果                                       |
| ---------------- | ---------------------------------------- |
| `unigram`        | 所有的特征 t1-t5 都一样。                         |
| `bigram`         | `[1, 2, 3, 4, 5]` 为 `[6, 7, 8, 8, 8]`。没有到达10的，效果不是很好。 |
| `trigram`        | 所有的特征 t1-t5 都一样。                         |
| `hashtag`        | 所有的特征 t1-t5 都一样。                         |
| `unigram_withrf` | 所有的特征 t1-t5 都一样。                         |
| `bigram_withrf`  | t2 的效果最好。                                |
| `trigram_withrf` | t1, t4 效果最好。                             |
| `hashtag_withrf` | t4, t5 最好。                               |

AdaBoost 总体都不太好，才0.17左右。

### 02. `DecisionTree`训练

| 特征               | 结果                              |      |
| ---------------- | ------------------------------- | ---- |
| `unigram`        |                                 |      |
| `bigram`         | t1-t5 效果都不错，最高可到 0.32           | *    |
| `trigram`        |                                 |      |
| `hashtag`        | 所有的特征 t1-t5 都一样，全10。都是 0.172    |      |
| `unigram_withrf` |                                 |      |
| `bigram_withrf`  |                                 |      |
| `trigram_withrf` |                                 |      |
| `hashtag_withrf` | [7, 7, 5, 6, 5]，效果都不太好，最高 0.172 |      |

### 03. `NaiveBayes` 训练

| 特征               | 结果               |
| ---------------- | ---------------- |
| `unigram`        |                  |
| `bigram`         |                  |
| `trigram`        |                  |
| `hashtag`        | 所有的特征 t1-t5 都一样。 |
| `unigram_withrf` |                  |
| `bigram_withrf`  |                  |
| `trigram_withrf` |                  |
| `hashtag_withrf` |                  |

## 2018-01-06

之前的评分系统中存在问题。目前全部重新修改。

增加的修改：修改 DictLoader，增加了RfLoader。增强系统性能。

具体内容看 /src/model_trainer中的修改内容。

## 2018-01-07

### 特征修改计划

* 使用 **wordsegment** 包对 hashtag 中连在一起的英文单词进行分割。

  hashtag之前在分割的时候存在问题。因为多个单词容易产生歧义。好在 **wordsegment** 包的语料库足够大，现在使用它来进行分词操作。

  (已经完成)

* 针对 tweet 中的 URL使用爬虫和 BeautifulSoup进行处理。

  (On going).

* ​


## 2018-01-08

### lambda 函数修正

今天我发现我之前使用的 lambda 函数都是错的。

**lambda** 函数原来的写法走：

```Python
nltk_unigram_withrf_t = {}
nltk_bigram_withrf_t = {}
nltk_trigram_withrf_t = {}
hashtag_t_withrf_t = {}
hashtag_unigram_withrf_t = {}
for __freq in range(1, 6):
    nltk_unigram_withrf_t[__freq] = lambda tweet: nltk_unigram_withrf_tu(tweet, __freq)
    nltk_unigram_withrf_t[__freq].__name__ = "nltk_unigram_withrf_t%d" % __freq
    nltk_bigram_withrf_t[__freq] = lambda tweet: nltk_bigram_withrf_tu(tweet, __freq)
    nltk_bigram_withrf_t[__freq].__name__ = "nltk_bigram_withrf_t%d" % __freq
    nltk_trigram_withrf_t[__freq] = lambda tweet: nltk_trigram_withrf_tu(tweet, __freq)
    nltk_trigram_withrf_t[__freq].__name__ = "nltk_trigram_withrf_t%d" % __freq
    hashtag_t_withrf_t[__freq] = lambda tweet: hashtag_withrf_tu(tweet, __freq)
    hashtag_t_withrf_t[__freq].__name__ = "hashtag_withrf_t%d" % __freq
    hashtag_unigram_withrf_t[__freq] = lambda tweet: hashtag_unigram_tu(tweet, __freq)
    hashtag_unigram_withrf_t[__freq].__name__ = "hashtag_unigram_withrf_t%d" % __freq
```

还有类似的写法也出现在 `src/model_loader/dict_loader.py`中。lambda中引用了外部的变量，因此这些变量在外部修改后也会跟着修改。因此所有计算出来的 Relation  Frequence 的都是基于原来的 Threshold 5 的数据计算的。

现在修改如下:

```Python
nltk_unigram_withrf_t = {}
nltk_bigram_withrf_t = {}
nltk_trigram_withrf_t = {}
hashtag_t_withrf_t = {}
hashtag_unigram_withrf_t = {}
for __freq in range(1, 6):
    nltk_unigram_withrf_t[__freq] = lambda tweet, __freq=__freq:\
        nltk_unigram_withrf_tu(tweet, __freq)
    nltk_unigram_withrf_t[__freq].__name__ = "nltk_unigram_withrf_t%d" % __freq
    nltk_bigram_withrf_t[__freq] = lambda tweet, __freq=__freq:\
        nltk_bigram_withrf_tu(tweet, __freq)
    nltk_bigram_withrf_t[__freq].__name__ = "nltk_bigram_withrf_t%d" % __freq
    nltk_trigram_withrf_t[__freq] = lambda tweet, __freq=__freq:\
        nltk_trigram_withrf_tu(tweet, __freq)
    nltk_trigram_withrf_t[__freq].__name__ = "nltk_trigram_withrf_t%d" % __freq
    hashtag_t_withrf_t[__freq] = lambda tweet, __freq=__freq: \
        hashtag_withrf_tu(tweet, __freq)
    hashtag_t_withrf_t[__freq].__name__ = "hashtag_withrf_t%d" % __freq
    hashtag_unigram_withrf_t[__freq] = lambda tweet, __freq=__freq: \
        hashtag_unigram_tu(tweet, __freq)
    hashtag_unigram_withrf_t[__freq].__name__ = "hashtag_unigram_withrf_t%d" % __freq
```

对 lambda 函数增加了一个可选参数 `__freq=__freq`，这样 `__freq` 变量的作用域就变成局部的变量了。

 这样一来，所有之前跑的结果以及 *rf* 文件全部要重新计算。

### 特征函数更新

这次更新过后，总共有如下的特征函数:

```Python
feature += [
    ners_existed,
    wv_google,
    wv_GloVe,
    sentilexi,
    emoticon,
    punction,
    elongated
]

for __freq in range(1, 6):
    feature.append(nltk_unigram_t[__freq])
    feature.append(nltk_bigram_t[__freq])
    feature.append(nltk_trigram_t[__freq])
    feature.append(hashtag_t[__freq])
    feature.append(hashtag_unigram_t[__freq])
    feature.append(nltk_unigram_withrf_t[__freq])
    feature.append(nltk_bigram_withrf_t[__freq])
    feature.append(nltk_trigram_withrf_t[__freq])
    feature.append(hashtag_t_withrf_t[__freq])
    feature.append(hashtag_unigram_withrf_t[__freq])
```

## 2018-01-13 ~ 2018-01-15

URL爬取工作:

### URL 爬取记录

对所有的 URL 进行原始链接跟踪，URL 的分布如下：

| 域名                            | 出现次数 |
| ----------------------------- | ---- |
| twitter.com                   | 320  |
| www.instagram.com             | 170  |
| www.youtube.com               | 48   |
| www.facebook.com              | 21   |
| www.tsu.co                    | 19   |
| hihid.co                      | 18   |
| realanaltube.com              | 12   |
| uk.reuters.com                | 11   |
| ift.tt                        | 11   |
| www.bbc.com                   | 9    |
| www.eureporter.co             | 9    |
| untappd.com                   | 8    |
| vine.co                       | 6    |
| pervaciousboutique.tumblr.com | 6    |
| ask.fm                        | 5    |
| www.washingtonpost.com        | 4    |
| mommasmoneymatters.com        | 4    |
| www.theguardian.com           | 4    |
| www.bbc.co.uk                 | 4    |
| www.theglobeandmail.com       | 3    |
| play.google.com               | 3    |
| reut.rs                       | 3    |
| www.myfairdaily.com           | 3    |
| koku.us                       | 3    |
| www.dailymail.co.uk           | 3    |
| www.reuters.com               | 3    |
| www.goodreads.com             | 3    |
| ww17.mydiycrafts.com          | 3    |
| www.telegraph.co.uk           | 2    |
| www.independent.co.uk         | 2    |
| timehop.com                   | 2    |
| www.seacretdirect.com         | 2    |
| 757l.tk                       | 2    |
| www.engadget.com              | 2    |
| parking.zunmi.cn              | 2    |
| rhiever.github.io             | 2    |
| www.etsy.com                  | 2    |
| soundcloud.com                | 2    |
| www.businessinsider.com       | 2    |
| pinterest.com                 | 2    |
| www.theage.com.au             | 2    |
| shirleyida.tumblr.com         | 2    |
| teameffortnetwork.biz         | 2    |
| bleacherreport.com            | 2    |

设定出现次数阈值为2，共计44条数据。在爬取原始网站的内容时，不同域名的网站，网站结构都不一样。这里对于排名靠前的网址进行了特判:

#### Twitter 处理

对指向 Twitter 的网站进行导航，有以下几种情况

* 页面指向一张照片，可能存在 Reviews。
* 页面不存在。
* 页面被主人保护，客人权限无法查看。
* 页面含有不安全元素，被Twitter查封。
* 页面指向的主人账号被查封，页面丢失。

针对以上的这些情况，我的处理方式为: 第一种，将页面中存在的 Reviews 进行抽取。其他全部当作 404 处理。

#### Instagram 处理

Instagram 仅存在两种情况

* 页面丢失
* 存在（并且含有评论）

Instagram正常的页面中，一定会存在一条评论，因为这是作者自己发送的内容（第一条评论，并且会被作为标题来处理）。

由于 Instagram 做过了反爬虫机制，所有的 class 都是随机的。因此我是用了 xpath 结合 css_selector 的方式，一层一层遍历，获取所有的评论信息。

爬取内容示例：

```JSON
"http://t.co/Lyr8HuLnSW": {
	"title": "I'm not usually a fan of anything #pumpkin or #pumpkinspiced but OMG. These are the best cookies I've ever had!!! They're good with vanilla frosting or without.",
	"current_url": "https://www.instagram.com/p/wc0gm4iVSj/",
	"review": [
		"I'm not usually a fan of anything #pumpkin or #pumpkinspiced but OMG. These are the best cookies I've ever had!!! They're good with vanilla frosting or without.",
		"I made these for Chad with the cream cheese frosting!",
		"I just used Vanilla and did it very sloppy because picky man Jacob doesn't like frosting. How is he considered American? They were sooooo good. @k_wimmer"
	]
},
```

#### Youtube 爬取处理

Youtube 的页面分为以下几种：

1. 视频页面被锁，或者丢失，无法继续打开。
2. 可以打开页面，并且含有评论。

Youtube 的页面中的评论因为使用了延迟加载机制，因此还要在爬取过程中，让网页自动向下滚动，将更多的内容加载出来，然后再爬取。代码比较技巧型，具体内容查看 `url_creeper.py` 。

爬取内容示例：

```JSON
"http://t.co/tp7MwdvwCW": {
    "title": "James Burke Connections\u00b3, Episode 7 A Special Place - YouTube",
    "current_url": "https://www.youtube.com/watch?v=qI2AkU3tR_Y&feature=youtu.be",
    "description": "",
    "review": [
        "bring James Burke back to TLC!",
        "vaca ! cows save lives",
        "Was it called Halley's comet in 1680 ?",
        "0:55 - 0:58 obviously watchin porn here.. can you guess what his hands are doing?",
        "ALERT; EPIC EYEBROW SPOTTED @13:45. THIS IS NOT A DRILL!",
        "James Burke is to history what Carl Sagan is to science."
    ]
}
```

#### Facebook 爬取处理

Facebook 爬取方式较为繁琐。首先，网站中部分内容的 class 使用了随机反爬虫机制。第二，因为爬虫访问页面时没有登录任何用户，因此会弹窗干扰爬虫。必须动态点击按钮去除弹窗才可以进步进行操作。

1. 获得主人发送的推文
2. 如果存在“评论按钮”，点击它
3. 如果存在“更多评论”，点击它
4. 可能有大部分评论被遮住了。如果存在“展开”按钮，则点击它。

具体的代码时比较技巧型的。具体内容查看 `url_creeper.py` 。

爬取内容示例:

```JSON
"http://t.co/5Gxuq6WDJk": {
    "title": "312 Dining Diva - What a pleasant way for Minneapolis'... | Facebook",
    "current_url": "https://www.facebook.com/photo.php?fbid=10152772316239584",
    "body": "What a pleasant way for Minneapolis' most prestigious publication to describe its female audience. #NOT",
    "review": [
        "Creative ..."
    ]
},
```

## 2018-01-16

