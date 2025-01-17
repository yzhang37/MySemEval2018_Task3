

# MySemEval2018_Task3 Experimental Logs

## 特征总结

### Liblinear

* unigram 使用 t2 作为候选的特征。
* bigram使用 t2 作为候选的特征。
* trigram 使用 t5 作为候选的特征。
* hashtag 使用 t2, t5 作为候选的特征 

- nltk_unigram_with_rf使用 t3 作为候选的特征。 
- nltk_bigram_with_rf使用 t2, t3 作为候选的特征。
- nltk_trigram_with_rf使用 t3 作为候选的特征。
- hashtag_with_rf使用 t3 作为候选的特征。

## Dec.29th 2017

今天整理了 liblinear, as following:

```Shell
Using following features:
==============================
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
nltk_unigram_t2
nltk_bigram_t2
nltk_trigram_t5
hashtag_t2
hashtag_t5
nltk_unigram_t3_with_rf
nltk_bigram_t2_with_rf
nltk_bigram_t3_with_rf
nltk_trigram_t3_with_rf
hashtag_t3_with_rf
```



## Dec.28th 2017

Hashtag on liblinear 测试

### 1. `hashtag on Liblinear`

#### 01

**0.0695715912621 hashtag_t1 | hashtag_t2 | hashtag_t5**

0.0695552300886 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.0688183991688 hashtag_t5

0.0684544190024 hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.0676752829008 hashtag_t2 | hashtag_t5

#### 02

**0.174607858187 hashtag_t2 | hashtag_t5**

0.174562920793 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.172710816083 hashtag_t5

0.172663783628 hashtag_t1 | hashtag_t2 | hashtag_t5

#### 03

**0.0790037701029 hashtag_t2 | hashtag_t5**

0.0783353034927 hashtag_t1 | hashtag_t2 | hashtag_t5

0.0779483959795 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.077512601315 hashtag_t5

#### 04

**0.0789197953858 hashtag_t2 | hashtag_t5**

0.0785576770143 hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.0785388076847 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.0785037612999 hashtag_t5

0.0784398149356 hashtag_t1 | hashtag_t2 | hashtag_t5

#### 05

**0.0710620879568 hashtag_t2 | hashtag_t5**

0.0700413369113 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.0684764852752 hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.068181150129 hashtag_t1 | hashtag_t2 | hashtag_t5

0.0672448324806 hashtag_t5

#### 06

**0.0690628026319 hashtag_t1 | hashtag_t2 | hashtag_t5**

0.0688974051928 hashtag_t5

0.0678254981616 hashtag_t2 | hashtag_t5

0.0671630972763 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

#### 07

**0.176947937651 hashtag_t2 | hashtag_t5**

0.176720499934 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.176701852222 hashtag_t1 | hashtag_t2 | hashtag_t5

0.175353307896 hashtag_t5

#### 08

**0.175531327689 hashtag_t1 | hashtag_t2 | hashtag_t5**

0.175103624217 hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.174776958237 hashtag_t5

0.174584894869 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.174075826208 hashtag_t2 | hashtag_t5

#### 09

**0.170801445334 hashtag_t1 | hashtag_t2 | hashtag_t5**

0.170479144997 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.170449332658 hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5

0.170051058111 hashtag_t5

0.169238945334 hashtag_t2 | hashtag_t5

#### 10

**0.0700327240258 hashtag_t3 | hashtag_t1 | hashtag_t4 | hashtag_t2 | hashtag_t5**

0.0686191224673 hashtag_t2 | hashtag_t5

0.0685872020893 hashtag_t1 | hashtag_t2 | hashtag_t5

0.0674177140936 hashtag_t5

#### hashtag_liblinear_hc 总结

| 名称         | 频次 (tier 1) |
| ---------- | ----------- |
| hashtag_t1 | 5           |
| hashtag_t2 | 10          |
| hashtag_t3 | 1           |
| hashtag_t4 | 1           |
| hashtag_t5 | 10          |

| 名称         | 频次 (tier 2) |
| ---------- | ----------- |
| hashtag_t1 | 13          |
| hashtag_t2 | 19          |
| hashtag_t3 | 6           |
| hashtag_t4 | 8           |
| hashtag_t5 | 20          |

经过统计，hashtag_t1 ~ hashtag_t5 分别为 `[5, 10, 1, 1, 10]`。因此 hashtag 使用 t2, t5 作为候选的特征。

### 2. `nltk_unigram_with_rf_on_liblinear` hc 测试

#### 01

**0.374886719357 nltk_unigram_t4_with_rf**

0.374760244988 nltk_unigram_t5_with_rf

0.374587315468 nltk_unigram_t3_with_rf

0.369904179085 nltk_unigram_t4_with_rf | nltk_unigram_t2_with_rf

0.369682744031 nltk_unigram_t4_with_rf | nltk_unigram_t3_with_rf

0.366806535373 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf

0.366795249877 nltk_unigram_t4_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.366606671791 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf

0.364977784252 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.364641353174 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t5_with_rf

0.363991968824 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 02

**0.37532266807 nltk_unigram_t1_with_rf**

0.375155779275 nltk_unigram_t2_with_rf

0.375017736947 nltk_unigram_t3_with_rf

0.365112443411 nltk_unigram_t1_with_rf | nltk_unigram_t3_with_rf

0.364119749494 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t3_with_rf

0.363762266448 nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.36314182279 nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t3_with_rf

0.360925222849 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t3_with_rf

0.360779139387 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.357863032185 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 03

**0.372243225234 nltk_unigram_t5_with_rf**

0.372081738213 nltk_unigram_t1_with_rf

0.372074030012 nltk_unigram_t3_with_rf

0.364753032539 nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.363827208861 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.362897225965 nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t3_with_rf

0.362739321187 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.362595526486 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t3_with_rf

0.361714056232 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 04

**0.382326827696 nltk_unigram_t4_with_rf**

0.382196923202 nltk_unigram_t2_with_rf

0.382023032419 nltk_unigram_t5_with_rf

0.378102302353 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf

0.371268827511 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf

0.371260916003 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf

0.371253426219 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.368501127488 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.368324097438 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t1_with_rf

0.361820681924 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 05

**0.370331444289 nltk_unigram_t3_with_rf**

0.365838135832 nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.365703455697 nltk_unigram_t4_with_rf | nltk_unigram_t3_with_rf

0.364959754993 nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.362941892923 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.362900150843 nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

0.361135455831 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.361002063282 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

0.357631672234 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

#### 06

**0.365341682765 nltk_unigram_t5_with_rf**

0.365259577918 nltk_unigram_t3_with_rf

0.365048941217 nltk_unigram_t4_with_rf

0.361126435042 nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.361087913481 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf

0.358203558755 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.358084503611 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.357926124432 nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.356136837949 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.355765308014 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.352492334585 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 07

**0.377537113745 nltk_unigram_t3_with_rf**

0.377280434151 nltk_unigram_t4_with_rf

0.364907912817 nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.364771476794 nltk_unigram_t4_with_rf | nltk_unigram_t3_with_rf

0.362477627289 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.362368449259 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.362339289822 nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t3_with_rf

0.359196390988 nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.359027558143 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.355364304892 nltk_unigram_t4_with_rf | nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 08

**0.372323398554 nltk_unigram_t2_with_rf**

0.372198682703 nltk_unigram_t5_with_rf

0.372020854044 nltk_unigram_t3_with_rf

0.361057364393 nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.35808082831 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

0.356868639049 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

0.356857141283 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.356249760706 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.356113529622 nltk_unigram_t4_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

#### 09

**0.365848629798 nltk_unigram_t3_with_rf**

0.365798171244 nltk_unigram_t1_with_rf

0.36563492937 nltk_unigram_t5_with_rf

0.36163827638 nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.359025689803 nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.35854884825 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.356717618489 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.356663521324 nltk_unigram_t4_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.355468918837 nltk_unigram_t1_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t4_with_rf

#### 10

**0.370733331342 nltk_unigram_t3_with_rf**

0.36589799807 nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.359597956477 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.359424850871 nltk_unigram_t4_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.358293765271 nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

0.358292402123 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf

0.355211837692 nltk_unigram_t4_with_rf | nltk_unigram_t5_with_rf | nltk_unigram_t2_with_rf | nltk_unigram_t3_with_rf | nltk_unigram_t1_with_rf

#### nltk_unigram_with_rf_liblinear_hc 总结

经过统计，nltk_unigram_with_rf_t1 ~ nltk_unigram_with_rf_t5 分别为 `[1, 1, 4, 2, 2]`。因此  使用 t3 作为候选的特征。

### 3. `nltk_bigram_with_rf_on_liblinear` hc 测试

#### 01
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 02
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 03
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 04
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 05
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 06
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 07
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 08
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 09
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### 10
0.3367500254 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336420418995 nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.336193697409 nltk_bigram_t3_with_rf
0.335798215714 nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t3_with_rf
0.334784588717 nltk_bigram_t4_with_rf | nltk_bigram_t5_with_rf | nltk_bigram_t2_with_rf | nltk_bigram_t1_with_rf | nltk_bigram_t3_with_rf

#### nltk_bigram_with_rf_hc 总结

| 名称                     | 频次 (2 tier) |
| ---------------------- | ----------- |
| nltk_bigram_t2_with_rf | 20          |
| nltk_bigram_t3_with_rf | 20          |
| nltk_bigram_t4_with_rf | 10          |
| nltk_bigram_t5_with_rf | 10          |

| 名称                     | 频次 （3 tier) |
| ---------------------- | ----------- |
| nltk_bigram_t2_with_rf | 20          |
| nltk_bigram_t3_with_rf | 30          |
| nltk_bigram_t4_with_rf | 10          |
| nltk_bigram_t5_with_rf | 10          |

经过统计，nltk_bigram_with_rf_t1 ~ nltk_bigram_with_rf_t5 分别为 `[0, 20, 30, 10, 10]`。因此  使用 t2, t3 作为候选的特征。

### 4. `nltk_trigram_with_rf_on_liblinear` hc 测试

#### 01
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 02
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 03
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 04
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 05
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 06
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 07
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 08
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 09
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### 10
**0.21637725752 nltk_trigram_t3_with_rf**
0.213552382858 nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212645767826 nltk_trigram_t4_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212564150491 nltk_trigram_t4_with_rf | nltk_trigram_t1_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf
0.212552604942 nltk_trigram_t4_with_rf | nltk_trigram_t5_with_rf | nltk_trigram_t2_with_rf | nltk_trigram_t3_with_rf

#### nltk_trigram_with_rf_liblinear_hc 总结

| 名称                      | 频次 (tier 1) |
| ----------------------- | ----------- |
| nltk_trigram_t3_with_rf | 10          |

经过统计，nltk_trigram_with_rf_t1 ~ nltk_trigram_with_rf_t5 分别为 `[0, 0, 10, 0, 0]`。因此使用 t3 作为候选的特征。

### 5. `hashtag_with_rf_on_liblinear` hc 测试

#### 01
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 02
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 03
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 04
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 05
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 06
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 07
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 08
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 09
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### 10
0.0780428468756 hashtag_t3_with_rf
0.0776298680755 hashtag_t4_with_rf | hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0775793741094 hashtag_t5_with_rf | hashtag_t1_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf
0.0770433910287 hashtag_t2_with_rf | hashtag_t3_with_rf
0.076906418647 hashtag_t5_with_rf | hashtag_t2_with_rf | hashtag_t3_with_rf

#### hashtag_with_rf_on_hc 总结

| 名称                 | 频次 (tier 1) |
| ------------------ | ----------- |
| hashtag_t3_with_rf | 10          |

| 名称                 | 频次 (tier 2) |
| ------------------ | ----------- |
| hashtag_t1_with_rf | 10          |
| hashtag_t2_with_rf | 10          |
| hashtag_t3_with_rf | 20          |
| hashtag_t4_with_rf | 10          |
| hashtag_t5_with_rf | 10          |

经过统计，使用 t3 作为候选的特征。

## Dec.27th 2017

bigram on liblinear测试。

### 1. `nltk_bigram`: hc 测试

测试算法: **liblinear**

#### 01

**0.329769282293 nltk_bigram_t1 | nltk_bigram_t2**

0.329519543919 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.327001651311 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.326181224416 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.315614780526 nltk_bigram_t2

#### 02

**0.325025886053 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2**

0.323133552669 nltk_bigram_t1 | nltk_bigram_t2

0.32228988636 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.318313253737 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.316268179702 nltk_bigram_t2

#### 03

**0.322274880314 nltk_bigram_t2**

0.318817306551 nltk_bigram_t1 | nltk_bigram_t2

0.316033852828 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.315886269771 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.315343400862 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

#### 04

**0.32957223487 nltk_bigram_t2**

0.329233138221 nltk_bigram_t3 | nltk_bigram_t5 | nltk_bigram_t4 | nltk_bigram_t1 | nltk_bigram_t2

0.328555781547 nltk_bigram_t5 | nltk_bigram_t4 | nltk_bigram_t1 | nltk_bigram_t2

0.328130951764 nltk_bigram_t1 | nltk_bigram_t2

0.32628454174 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

#### 05

**0.333845714189 nltk_bigram_t5 | nltk_bigram_t4 | nltk_bigram_t1 | nltk_bigram_t2**

0.333795434796 nltk_bigram_t5 | nltk_bigram_t4 | nltk_bigram_t3 | nltk_bigram_t1 | nltk_bigram_t2

0.327578802962 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.326570003365 nltk_bigram_t1 | nltk_bigram_t2

0.325213642235 nltk_bigram_t2

#### 06

**0.33144529914 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2**

0.330553544644 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.328270687501 nltk_bigram_t5 | nltk_bigram_t2

0.32782777866 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t2

0.326036112671 nltk_bigram_t2

#### 07

**0.323809158553 nltk_bigram_t2**

0.323180676245 nltk_bigram_t1 | nltk_bigram_t2

0.322398858757 nltk_bigram_t3 | nltk_bigram_t5 | nltk_bigram_t4 | nltk_bigram_t1 | nltk_bigram_t2

0.322274062136 nltk_bigram_t5 | nltk_bigram_t4 | nltk_bigram_t1 | nltk_bigram_t2

0.320769090866 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

#### 08

**0.324599259469 nltk_bigram_t2**

0.321102737864 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.320918672177 nltk_bigram_t1 | nltk_bigram_t2

0.320075363555 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.319275515587 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

#### 09

**0.338108022053 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2**

0.335639159754 nltk_bigram_t1 | nltk_bigram_t2

0.335555606867 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.334206596761 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.331090775257 nltk_bigram_t2

#### 10

**0.318037136028 nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2**

0.317953429582 nltk_bigram_t1 | nltk_bigram_t2

0.317691036496 nltk_bigram_t2

0.317322420236 nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

0.315766913735 nltk_bigram_t3 | nltk_bigram_t4 | nltk_bigram_t5 | nltk_bigram_t1 | nltk_bigram_t2

#### nltk_bigram_liblinear_hc 总结

经过统计，bigram_t1 ~ trigram_t5 分别为 `[6, 10, 1, 3, 5]`。因此 bigram 使用 t2 作为候选的特征。

### 2. `nltk_trigram` hc 测试

#### 01

**0.23921504211 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5**

0.23875786777 nltk_trigram_t1 | nltk_trigram_t5

0.23858578281 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.23760898914 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

0.237313149475 nltk_trigram_t5

#### 02

**0.21531314499 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5**

0.215063462487 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

0.214988073947 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.214832100699 nltk_trigram_t1 | nltk_trigram_t5

0.212635959908 nltk_trigram_t5

#### 03

**0.189942715769 nltk_trigram_t5**

0.189262400944 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.189095106819 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.189057091945 nltk_trigram_t1 | nltk_trigram_t5

0.188931979076 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

#### 04

**0.217262918807 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5**

0.216393445735 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

0.216266403657 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.215500431585 nltk_trigram_t1 | nltk_trigram_t5

0.215338895292 nltk_trigram_t5

#### 05

**0.245088801151 nltk_trigram_t1 | nltk_trigram_t5**

0.243855163486 nltk_trigram_t5

0.243226036116 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.242798445602 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.242705958982 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

#### 06

**0.242128330495 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5**

0.241958584628 nltk_trigram_t1 | nltk_trigram_t5

0.241859742675 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.241469167599 nltk_trigram_t5

0.241276111254 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

#### 07

**0.188404413349 nltk_trigram_t5**

0.188385367681 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.18805421648 nltk_trigram_t1 | nltk_trigram_t5

0.187920133426 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.187547654894 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

#### 08

**0.237856024127 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5**

0.237720830862 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.237685641866 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

0.237424702995 nltk_trigram_t1 | nltk_trigram_t5

0.235815049575 nltk_trigram_t5

#### 09

**0.190378263048 nltk_trigram_t5**

0.189901484901 nltk_trigram_t1 | nltk_trigram_t5

0.189503052049 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.188452711737 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

0.188429328751 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

#### 10

**0.176093214229 nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5**

0.175690703603 nltk_trigram_t1 | nltk_trigram_t5

0.175594391871 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t4 | nltk_trigram_t1 | nltk_trigram_t5

0.175045287593 nltk_trigram_t2 | nltk_trigram_t3 | nltk_trigram_t1 | nltk_trigram_t5

0.173327687163 nltk_trigram_t5

#### nltk_trigram_liblinear_hc 总结

经过统计，trigram_t1 ~ trigram_t5 分别为 `[7, 2, 6, 0, 10]`。因此 trigram 使用 t5 作为候选的特征。

## Dec.25th 2017

很久都没有跑过这个程序了。最近集中在于学习深度学习的论文。

觉得上次程序跑的有问题，一次性同时使用了太多的特征，一切计算hc，而且每个特征都有较大的随机性。因此最终计算结果不可行所以准备如下：

### 1. nltk_unigram: hc 测试

测试算法：**liblinear**

#### 01

**0.369581985754 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t2**

**0.369448070684 nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2**

0.369314621888 nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t2

0.369291312663 nltk_unigram_t1 | nltk_unigram_t2

0.369154192201 nltk_unigram_t5 | nltk_unigram_t2

0.364930074987 nltk_unigram_t2

0.364624855942 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t2

0.363834142979 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.362434469128 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 02

**0.371875128179 nltk_unigram_t1 | nltk_unigram_t5**

**0.371698587807 nltk_unigram_t4 | nltk_unigram_t5**

0.371457744993 nltk_unigram_t5 | nltk_unigram_t2

0.369604307925 nltk_unigram_t5

0.369498068712 nltk_unigram_t2

0.365035861433 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5

0.3649854524 nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.362983011793 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5

0.362836743097 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.361204970356 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 03

**0.378551644174 nltk_unigram_t5**

**0.378537625253 nltk_unigram_t2**

0.376581015591 nltk_unigram_t5 | nltk_unigram_t2

0.376460533664 nltk_unigram_t3 | nltk_unigram_t5

0.376350196061 nltk_unigram_t4 | nltk_unigram_t5

0.368861264512 nltk_unigram_t3 | nltk_unigram_t5 | nltk_unigram_t2

0.368723646859 nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.367847871397 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.36742745158 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t5 | nltk_unigram_t2

0.367402778682 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 04

**0.384446063772 nltk_unigram_t2**

**0.381780317057 nltk_unigram_t5 | nltk_unigram_t2**

0.381647864134 nltk_unigram_t1 | nltk_unigram_t2

0.37881348899 nltk_unigram_t3 | nltk_unigram_t5 | nltk_unigram_t2

0.373945985459 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.373805151317 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t5 | nltk_unigram_t2

0.373744545524 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 05

**0.37026796411 nltk_unigram_t2**

**0.369977250356 nltk_unigram_t5 | nltk_unigram_t2**

0.369934148556 nltk_unigram_t3 | nltk_unigram_t2

0.369095980008 nltk_unigram_t1 | nltk_unigram_t2

0.367779134053 nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.367584153752 nltk_unigram_t3 | nltk_unigram_t5 | nltk_unigram_t2

0.364725930448 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.362115045598 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 06

**0.369902891015 nltk_unigram_t1**

**0.369722940693 nltk_unigram_t2**

0.364761041647 nltk_unigram_t4 | nltk_unigram_t1

0.364628973125 nltk_unigram_t1 | nltk_unigram_t2

0.364489311017 nltk_unigram_t3 | nltk_unigram_t1

0.364357253849 nltk_unigram_t1 | nltk_unigram_t5

0.362257064608 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1

0.360817348327 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t2

0.360813688964 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5

0.360682557976 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 07

**0.375362225114 nltk_unigram_t2**

**0.372375993269 nltk_unigram_t5 | nltk_unigram_t2**

0.372357146321 nltk_unigram_t3 | nltk_unigram_t2

0.372310860566 nltk_unigram_t1 | nltk_unigram_t2

0.369765685089 nltk_unigram_t3 | nltk_unigram_t5 | nltk_unigram_t2

0.367632057585 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.365150553079 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 08

**0.375834108272 nltk_unigram_t2**

**0.375816681375 nltk_unigram_t4**

0.37566087279 nltk_unigram_t3

0.373470633425 nltk_unigram_t4 | nltk_unigram_t2

0.37329499045 nltk_unigram_t5 | nltk_unigram_t2

0.370375810961 nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t2

0.370246721527 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t2

0.368326229539 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t2

0.367914889534 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 09

**0.375032610065 nltk_unigram_t2**

**0.374024245252 nltk_unigram_t5 | nltk_unigram_t2**

0.373852091069 nltk_unigram_t1 | nltk_unigram_t2

0.366304327541 nltk_unigram_t3 | nltk_unigram_t5 | nltk_unigram_t2

0.364291810749 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.363968649007 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t5 | nltk_unigram_t2

0.362164126306 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### 10

**0.369668086845 nltk_unigram_t2**

**0.368103152901 nltk_unigram_t3 | nltk_unigram_t2**

0.368102587093 nltk_unigram_t4 | nltk_unigram_t2

0.367833087522 nltk_unigram_t1 | nltk_unigram_t2

0.367805740817 nltk_unigram_t5 | nltk_unigram_t2

0.365517123005 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

0.36538299754 nltk_unigram_t3 | nltk_unigram_t5 | nltk_unigram_t2

0.365356303928 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t5 | nltk_unigram_t2

0.365247164172 nltk_unigram_t3 | nltk_unigram_t1 | nltk_unigram_t2

0.360248885772 nltk_unigram_t3 | nltk_unigram_t4 | nltk_unigram_t1 | nltk_unigram_t5 | nltk_unigram_t2

#### nltk_unigram on `liblinear` 总结

经过统计，unigram_t1 ~ ungram_t5 分别为 `[3, 7, 1, 0, 2]`。因此 unigram 使用 t2 作为候选的特征。

## Dec. 13th 2017

本次的任务:

1. 将unigram, bigram, trigram等所有的词典，按照t1-t5分开计算，形成几个独立的特征。
2. 分别对于所有的词典，需要计算rf的这些部分，分别按照任务A，B拆开成单独的文件。
3. 对所有的t, 或者a,b设计成单独的特征。



PS:更新了以后，总共有这么多的特征函数了！

```
Using following features:
==============================
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
nltk_unigram_t1
nltk_bigram_t1
nltk_trigram_t1
hashtag_t1
nltk_unigram_t1_with_rf
nltk_bigram_t1_with_rf
nltk_trigram_t1_with_rf
hashtag_t1_with_rf
nltk_unigram_t2
nltk_bigram_t2
nltk_trigram_t2
hashtag_t2
nltk_unigram_t2_with_rf
nltk_bigram_t2_with_rf
nltk_trigram_t2_with_rf
hashtag_t2_with_rf
nltk_unigram_t3
nltk_bigram_t3
nltk_trigram_t3
hashtag_t3
nltk_unigram_t3_with_rf
nltk_bigram_t3_with_rf
nltk_trigram_t3_with_rf
hashtag_t3_with_rf
nltk_unigram_t4
nltk_bigram_t4
nltk_trigram_t4
hashtag_t4
nltk_unigram_t4_with_rf
nltk_bigram_t4_with_rf
nltk_trigram_t4_with_rf
hashtag_t4_with_rf
nltk_unigram_t5
nltk_bigram_t5
nltk_trigram_t5
hashtag_t5
nltk_unigram_t5_with_rf
nltk_bigram_t5_with_rf
nltk_trigram_t5_with_rf
hashtag_t5_with_rf
==============================
```






## Dec. 12th 2017
### 计算1

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
nltk_trigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_trigram_t1 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 45.07%     | 40.85%     | 41.48%     |
| Fold 2   | 43.88%     | 41.27%     | 41.30%     |
| Fold 3   | 43.52%     | 39.75%     | 39.77%     |
| Fold 4   | 48.02%     | 43.27%     | 44.07%     |
| Fold 5   | 51.22%     | 42.76%     | 44.45%     |
| Fold 6   | 39.07%     | 38.55%     | 38.12%     |
| Fold 7   | 41.96%     | 41.43%     | 41.41%     |
| Fold 8   | 47.93%     | 39.73%     | 40.15%     |
| Fold 9   | 39.76%     | 39.20%     | 38.73%     |
| Fold 10  | 49.41%     | 42.89%     | 44.03%     |
| **Mean** | **44.98%** | **40.97%** | **41.35%** |

#### Train Result Table: 增加了nltk_trigram_t1 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 43.54%     | 40.75%     | 41.25%     |
| Fold 2   | 42.65%     | 41.47%     | 41.48%     |
| Fold 3   | 50.90%     | 40.85%     | 42.04%     |
| Fold 4   | 36.32%     | 36.67%     | 35.80%     |
| Fold 5   | 55.44%     | 42.59%     | 43.52%     |
| Fold 6   | 44.10%     | 40.67%     | 40.99%     |
| Fold 7   | 41.46%     | 39.68%     | 39.45%     |
| Fold 8   | 43.77%     | 40.68%     | 40.67%     |
| Fold 9   | 52.08%     | 43.34%     | 44.20%     |
| Fold 10  | 42.08%     | 37.38%     | 37.68%     |
| **Mean** | **45.23%** | **40.41%** | **40.71%** |

#### Train Result Table: 增加了nltk_trigram_t1 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 35.28%     | 35.64%     | 34.17%     |
| Fold 2   | 47.42%     | 45.42%     | 45.85%     |
| Fold 3   | 40.65%     | 40.05%     | 39.61%     |
| Fold 4   | 53.05%     | 40.12%     | 41.30%     |
| Fold 5   | 48.54%     | 40.66%     | 41.56%     |
| Fold 6   | 46.12%     | 40.69%     | 41.46%     |
| Fold 7   | 43.98%     | 39.71%     | 39.97%     |
| Fold 8   | 58.00%     | 47.04%     | 49.68%     |
| Fold 9   | 55.11%     | 40.81%     | 42.46%     |
| Fold 10  | 40.96%     | 38.02%     | 38.05%     |
| **Mean** | **46.91%** | **40.82%** | **41.41%** |

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
nltk_trigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```
#### Train Result Table: 增加了nltk_trigram_t1_rf


| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 39.44%     | 38.75%     | 38.55%     |
| Fold 2   | 59.82%     | 38.10%     | 37.69%     |
| Fold 3   | 58.15%     | 40.51%     | 41.88%     |
| Fold 4   | 49.84%     | 40.91%     | 41.39%     |
| Fold 5   | 43.33%     | 39.80%     | 39.94%     |
| Fold 6   | 54.73%     | 41.98%     | 43.24%     |
| Fold 7   | 51.49%     | 40.76%     | 41.96%     |
| Fold 8   | 45.47%     | 41.62%     | 41.97%     |
| Fold 9   | 47.80%     | 39.95%     | 40.24%     |
| Fold 10  | 42.60%     | 40.61%     | 40.42%     |
| **Mean** | **49.27%** | **40.30%** | **40.73%** |

#### Train Result Table: 增加了nltk_trigram_t1_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 41.59%     | 39.99%     | 39.56%     |
| Fold 2   | 36.39%     | 37.00%     | 35.64%     |
| Fold 3   | 40.68%     | 39.96%     | 39.78%     |
| Fold 4   | 38.70%     | 38.60%     | 38.23%     |
| Fold 5   | 38.33%     | 38.36%     | 37.09%     |
| Fold 6   | 74.65%     | 47.85%     | 49.13%     |
| Fold 7   | 58.59%     | 42.38%     | 44.14%     |
| Fold 8   | 52.95%     | 38.87%     | 39.38%     |
| Fold 9   | 46.48%     | 42.09%     | 42.49%     |
| Fold 10  | 42.72%     | 38.16%     | 37.61%     |
| **Mean** | **47.11%** | **40.33%** | **40.31%** |

#### 结论

trigram_t1的f1平均值并不是很好，但是也有个别例子特别高，达到了49.13%。

### 计算2

计算在原有的基础上，加上 nltk_trigram_t2的效果

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
nltk_trigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```
#### Train Result Table: 增加了nltk_trigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 43.04%     | 37.79%     | 38.50%     |
| Fold 2   | 47.08%     | 42.59%     | 43.48%     |
| Fold 3   | 46.35%     | 41.13%     | 42.18%     |
| Fold 4   | 47.31%     | 41.06%     | 42.51%     |
| Fold 5   | 39.23%     | 40.08%     | 39.59%     |
| Fold 6   | 45.69%     | 39.30%     | 40.07%     |
| Fold 7   | 57.89%     | 49.19%     | 50.95%     |
| Fold 8   | 58.89%     | 47.43%     | 49.32%     |
| Fold 9   | 54.35%     | 44.06%     | 45.92%     |
| Fold 10  | 43.71%     | 38.57%     | 39.58%     |
| **Mean** | **48.36%** | **42.12%** | **43.21%** |

#### Train Result Table: 增加了nltk_trigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 41.58%     | 39.15%     | 39.14%     |
| Fold 2   | 45.32%     | 41.40%     | 42.23%     |
| Fold 3   | 51.12%     | 40.71%     | 42.03%     |
| Fold 4   | 46.01%     | 42.11%     | 42.66%     |
| Fold 5   | 40.43%     | 39.80%     | 39.62%     |
| Fold 6   | 48.67%     | 38.78%     | 39.69%     |
| Fold 7   | 42.50%     | 39.74%     | 40.11%     |
| Fold 8   | 54.46%     | 43.11%     | 44.90%     |
| Fold 9   | 45.24%     | 40.79%     | 41.42%     |
| Fold 10  | 65.04%     | 40.35%     | 40.98%     |
| **Mean** | **48.04%** | **40.59%** | **41.28%** |

#### Train Result Table: 增加了nltk_trigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 62.90%     | 45.20%     | 48.70%     |
| Fold 2   | 58.52%     | 44.13%     | 46.42%     |
| Fold 3   | 46.78%     | 39.28%     | 40.14%     |
| Fold 4   | 42.46%     | 42.42%     | 42.22%     |
| Fold 5   | 42.42%     | 39.73%     | 40.10%     |
| Fold 6   | 43.90%     | 41.34%     | 41.69%     |
| Fold 7   | 44.63%     | 40.03%     | 39.99%     |
| Fold 8   | 42.26%     | 39.39%     | 39.65%     |
| Fold 9   | 45.68%     | 39.58%     | 39.92%     |
| Fold 10  | 42.13%     | 41.09%     | 41.10%     |
| **Mean** | **47.17%** | **41.22%** | **41.99%** |

#### Train Result Table: 增加了nltk_trigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 41.74%     | 41.46%     | 41.24%     |
| Fold 2   | 41.43%     | 38.73%     | 39.49%     |
| Fold 3   | 44.42%     | 41.30%     | 41.75%     |
| Fold 4   | 65.92%     | 41.22%     | 43.27%     |
| Fold 5   | 50.69%     | 45.44%     | 46.40%     |
| Fold 6   | 37.80%     | 38.29%     | 37.56%     |
| Fold 7   | 46.56%     | 39.36%     | 40.56%     |
| Fold 8   | 50.05%     | 42.17%     | 43.65%     |
| Fold 9   | 41.31%     | 40.73%     | 40.59%     |
| Fold 10  | 57.55%     | 42.08%     | 43.26%     |
| **Mean** | **47.75%** | **41.08%** | **41.78%** |

nltk_trigram_t2 平均都在 42%左右，在原基础上的效果。

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
nltk_trigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_trigram_t2_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 50.25%     | 42.51%     | 43.92%     |
| Fold 2   | 48.28%     | 41.32%     | 42.59%     |
| Fold 3   | 50.27%     | 38.34%     | 39.12%     |
| Fold 4   | 44.69%     | 43.81%     | 43.95%     |
| Fold 5   | 45.73%     | 41.41%     | 42.01%     |
| Fold 6   | 54.36%     | 42.34%     | 44.08%     |
| Fold 7   | 38.77%     | 37.95%     | 37.78%     |
| Fold 8   | 52.56%     | 40.74%     | 41.44%     |
| Fold 9   | 42.77%     | 39.98%     | 40.34%     |
| Fold 10  | 41.86%     | 40.18%     | 40.41%     |
| **Mean** | **46.95%** | **40.86%** | **41.56%** |

#### Train Result Table: 增加了nltk_trigram_t2_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 54.78%     | 38.18%     | 40.15%     |
| Fold 2   | 45.00%     | 43.13%     | 43.28%     |
| Fold 3   | 47.42%     | 43.90%     | 44.76%     |
| Fold 4   | 54.54%     | 42.17%     | 44.24%     |
| Fold 5   | 37.94%     | 37.61%     | 37.19%     |
| Fold 6   | 47.57%     | 40.66%     | 41.62%     |
| Fold 7   | 35.54%     | 37.52%     | 35.98%     |
| Fold 8   | 47.28%     | 42.25%     | 42.73%     |
| Fold 9   | 44.15%     | 40.20%     | 40.99%     |
| Fold 10  | 44.36%     | 41.42%     | 41.67%     |
| **Mean** | **45.86%** | **40.70%** | **41.26%** |

#### Train Result Table: 增加了nltk_trigram_t2_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 51.48%     | 41.06%     | 41.58%     |
| Fold 2   | 47.18%     | 40.29%     | 40.94%     |
| Fold 3   | 44.21%     | 39.92%     | 40.63%     |
| Fold 4   | 42.60%     | 40.64%     | 40.87%     |
| Fold 5   | 49.98%     | 40.18%     | 41.03%     |
| Fold 6   | 57.96%     | 42.32%     | 44.79%     |
| Fold 7   | 47.05%     | 42.32%     | 42.88%     |
| Fold 8   | 39.58%     | 39.65%     | 39.31%     |
| Fold 9   | 50.80%     | 46.53%     | 47.64%     |
| Fold 10  | 44.15%     | 39.49%     | 39.78%     |
| **Mean** | **47.50%** | **41.24%** | **41.95%** |

 `bigram_t2_rf` +`trigram_t2_rf` 的平均值只有 41.59%，比不上不加它的 `bigram_t2_rf` .

### 计算3

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
nltk_trigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table:  增加了nltk_trigram_t3 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 36.29%     | 37.64%     | 36.54%     |
| Fold 2   | 38.54%     | 38.64%     | 38.19%     |
| Fold 3   | 44.26%     | 41.29%     | 41.65%     |
| Fold 4   | 43.22%     | 41.64%     | 41.96%     |
| Fold 5   | 49.23%     | 42.21%     | 43.89%     |
| Fold 6   | 53.83%     | 42.14%     | 42.78%     |
| Fold 7   | 50.94%     | 42.45%     | 44.52%     |
| Fold 8   | 43.85%     | 40.06%     | 41.00%     |
| Fold 9   | 45.68%     | 42.67%     | 43.01%     |
| Fold 10  | 42.19%     | 37.88%     | 38.40%     |
| **Mean** | **44.80%** | **40.66%** | **41.19%** |

#### Train Result Table:  增加了nltk_trigram_t3 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 44.23%     | 40.58%     | 41.01%     |
| Fold 2   | 50.52%     | 41.77%     | 43.35%     |
| Fold 3   | 37.02%     | 37.31%     | 36.76%     |
| Fold 4   | 54.39%     | 43.67%     | 45.69%     |
| Fold 5   | 50.71%     | 43.84%     | 45.28%     |
| Fold 6   | 45.11%     | 39.05%     | 39.94%     |
| Fold 7   | 43.85%     | 39.88%     | 39.95%     |
| Fold 8   | 47.34%     | 42.83%     | 43.90%     |
| Fold 9   | 59.59%     | 42.50%     | 44.20%     |
| Fold 10  | 49.10%     | 42.65%     | 43.63%     |
| **Mean** | **48.19%** | **41.41%** | **42.37%** |

#### Train Result Table: 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 55.88%     | 39.89%     | 40.90%     |
| Fold 2   | 43.64%     | 41.32%     | 41.60%     |
| Fold 3   | 39.49%     | 38.47%     | 38.14%     |
| Fold 4   | 51.89%     | 43.61%     | 45.05%     |
| Fold 5   | 40.20%     | 40.94%     | 40.05%     |
| Fold 6   | 40.70%     | 39.19%     | 39.27%     |
| Fold 7   | 40.73%     | 36.84%     | 36.98%     |
| Fold 8   | 40.60%     | 39.03%     | 39.27%     |
| Fold 9   | 48.37%     | 40.92%     | 42.00%     |
| Fold 10  | 40.16%     | 39.43%     | 39.35%     |
| **Mean** | **44.17%** | **39.96%** | **40.26%** |

#### Train Result Table: 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 49.14%     | 40.92%     | 42.09%     |
| Fold 2   | 37.44%     | 38.16%     | 37.53%     |
| Fold 3   | 51.24%     | 43.21%     | 43.78%     |
| Fold 4   | 47.50%     | 39.33%     | 40.89%     |
| Fold 5   | 48.68%     | 40.81%     | 42.16%     |
| Fold 6   | 39.68%     | 39.13%     | 39.01%     |
| Fold 7   | 58.80%     | 47.73%     | 50.88%     |
| Fold 8   | 42.44%     | 42.02%     | 41.82%     |
| Fold 9   | 42.95%     | 38.89%     | 39.33%     |
| Fold 10  | 51.08%     | 44.07%     | 45.56%     |
| **Mean** | **46.90%** | **41.43%** | **42.30%** |

#### Train Result Table: 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 39.21%     | 38.49%     | 38.35%     |
| Fold 2   | 41.64%     | 40.54%     | 40.59%     |
| Fold 3   | 37.81%     | 38.69%     | 38.10%     |
| Fold 4   | 42.22%     | 38.32%     | 38.31%     |
| Fold 5   | 45.45%     | 39.14%     | 40.14%     |
| Fold 6   | 44.41%     | 42.52%     | 42.84%     |
| Fold 7   | 59.06%     | 41.90%     | 43.67%     |
| Fold 8   | 45.84%     | 40.33%     | 40.85%     |
| Fold 9   | 40.41%     | 37.58%     | 37.90%     |
| Fold 10  | 50.13%     | 42.11%     | 43.07%     |
| **Mean** | **44.62%** | **39.96%** | **40.38%** |

### 结论:

nltk_trigram_t3效果不好，rf版本不稳定。不能考虑使用。

## Dec. 11th 2017

### 计算1
首先，我发现之前写的抽特征的函数 nltk_unigram_with_rf, nltk_unigram 调用的是 unigram，不是 nltk_unigram。因此会有一定问题，修改后我重新生成了 nltk_unigram 的词典，并计算了 rf。训练结果如下:

```
Using following features:
==============================
nltk_unigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 第一次跑 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 50.46%     | 41.04%     | 42.10%     |
| Fold 2   | 48.04%     | 40.88%     | 41.90%     |
| Fold 3   | 51.63%     | 43.82%     | 45.71%     |
| Fold 4   | 48.52%     | 44.48%     | 45.55%     |
| Fold 5   | 39.79%     | 38.94%     | 38.64%     |
| Fold 6   | 34.49%     | 36.14%     | 34.50%     |
| Fold 7   | 55.48%     | 43.29%     | 43.90%     |
| Fold 8   | 52.55%     | 39.58%     | 40.17%     |
| Fold 9   | 39.99%     | 38.87%     | 39.11%     |
| Fold 10  | 42.03%     | 38.97%     | 39.57%     |
| **Mean** | **46.30%** | **40.60%** | **41.11%** |

#### Train Result Table: 第二次跑 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 38.84%     | 39.41%     | 39.04%     |
| Fold 2   | 43.19%     | 39.85%     | 40.40%     |
| Fold 3   | 37.65%     | 37.47%     | 36.61%     |
| Fold 4   | 51.67%     | 42.74%     | 44.33%     |
| Fold 5   | 43.85%     | 39.10%     | 39.40%     |
| Fold 6   | 43.34%     | 39.49%     | 39.87%     |
| Fold 7   | 40.80%     | 37.64%     | 37.97%     |
| Fold 8   | 45.35%     | 41.76%     | 42.52%     |
| Fold 9   | 48.03%     | 43.94%     | 44.98%     |
| Fold 10  | 53.22%     | 40.96%     | 42.39%     |
| **Mean** | **44.59%** | **40.24%** | **40.75%** |

平均F1接近41%

### 计算2

添加 nltk_bigram (freq 1), 分别测试 nltk_bigram / nltk_bigram_rf。

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t1 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 43.56%     | 38.03%     | 37.94%     |
| Fold 2   | 49.75%     | 42.27%     | 43.01%     |
| Fold 3   | 52.88%     | 40.11%     | 41.10%     |
| Fold 4   | 47.12%     | 40.99%     | 41.20%     |
| Fold 5   | 47.18%     | 40.78%     | 41.49%     |
| Fold 6   | 54.20%     | 44.12%     | 46.32%     |
| Fold 7   | 47.49%     | 38.86%     | 39.59%     |
| Fold 8   | 50.18%     | 41.54%     | 42.37%     |
| Fold 9   | 45.15%     | 39.81%     | 40.96%     |
| Fold 10  | 53.11%     | 39.80%     | 40.95%     |
| **Mean** | **49.06%** | **40.63%** | **41.49%** |

#### Train Result Table: 增加了nltk_bigram_t1 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 40.47%     | 36.84%     | 36.74%     |
| Fold 2   | 44.97%     | 39.71%     | 39.79%     |
| Fold 3   | 47.35%     | 40.26%     | 40.63%     |
| Fold 4   | 44.60%     | 41.21%     | 40.97%     |
| Fold 5   | 42.86%     | 40.37%     | 40.63%     |
| Fold 6   | 51.47%     | 42.77%     | 43.42%     |
| Fold 7   | 40.23%     | 38.87%     | 39.06%     |
| Fold 8   | 46.03%     | 39.21%     | 39.77%     |
| Fold 9   | 50.30%     | 40.61%     | 42.01%     |
| Fold 10  | 47.36%     | 42.57%     | 43.56%     |
| **Mean** | **45.56%** | **40.24%** | **40.66%** |

#### Train Result Table: 增加了nltk_bigram_t1 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 39.20%     | 38.08%     | 37.50%     |
| Fold 2   | 40.96%     | 39.00%     | 39.16%     |
| Fold 3   | 44.41%     | 39.53%     | 40.31%     |
| Fold 4   | 61.42%     | 44.75%     | 46.12%     |
| Fold 5   | 49.87%     | 41.41%     | 42.43%     |
| Fold 6   | 50.80%     | 40.13%     | 40.90%     |
| Fold 7   | 41.75%     | 40.45%     | 40.18%     |
| Fold 8   | 44.40%     | 39.35%     | 39.88%     |
| Fold 9   | 49.97%     | 39.77%     | 40.14%     |
| Fold 10  | 44.86%     | 42.31%     | 42.64%     |
| **Mean** | **46.76%** | **40.48%** | **40.92%** |

结论：没有太大的变化。

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t1_with_rf 

| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 41.01%     | 40.07%     | 39.51%     |
| Fold 2   | 66.50%     | 42.73%     | 44.15%     |
| Fold 3   | 62.16%     | 44.54%     | 46.26%     |
| Fold 4   | 42.14%     | 39.59%     | 39.95%     |
| Fold 5   | 53.15%     | 40.20%     | 40.66%     |
| Fold 6   | 41.17%     | 40.33%     | 38.83%     |
| Fold 7   | 39.91%     | 37.47%     | 37.14%     |
| Fold 8   | 52.41%     | 41.80%     | 42.94%     |
| Fold 9   | 58.93%     | 42.64%     | 44.58%     |
| Fold 10  | 52.85%     | 41.62%     | 42.91%     |
| **Mean** | **51.02%** | **41.10%** | **41.69%** |

#### Train Result Table: 增加了nltk_bigram_t1_with_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 69.00%     | 43.46%     | 44.49%     |
| Fold 2   | 62.48%     | 37.79%     | 38.24%     |
| Fold 3   | 48.07%     | 43.45%     | 44.34%     |
| Fold 4   | 63.30%     | 42.98%     | 45.55%     |
| Fold 5   | 44.81%     | 41.05%     | 40.57%     |
| Fold 6   | 39.40%     | 38.12%     | 37.49%     |
| Fold 7   | 49.46%     | 41.73%     | 42.67%     |
| Fold 8   | 49.11%     | 41.77%     | 42.72%     |
| Fold 9   | 40.45%     | 39.04%     | 38.69%     |
| Fold 10  | 47.71%     | 39.07%     | 39.40%     |
| **Mean** | **51.38%** | **40.85%** | **41.41%** |

#### Train Result Table: 增加了nltk_bigram_t1_with_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 39.14%     | 38.47%     | 37.64%     |
| Fold 2   | 48.29%     | 41.71%     | 42.52%     |
| Fold 3   | 60.87%     | 41.11%     | 42.71%     |
| Fold 4   | 47.81%     | 41.90%     | 42.94%     |
| Fold 5   | 34.95%     | 35.56%     | 33.97%     |
| Fold 6   | 53.00%     | 42.80%     | 44.52%     |
| Fold 7   | 41.53%     | 40.69%     | 40.48%     |
| Fold 8   | 39.18%     | 38.75%     | 38.47%     |
| Fold 9   | 60.54%     | 45.70%     | 47.38%     |
| Fold 10  | 49.95%     | 41.98%     | 42.68%     |
| **Mean** | **47.53%** | **40.87%** | **41.33%** |

#### 结论

使用了 nltk_bigram_t1_with_rf后，F1平均值到达了41.33%以上。

修改的文件:

dict_loader.py

dict_creator.py


### 计算 3

添加 nltk_bigram (freq 2), 分别测试 nltk_bigram / nltk_bigram_rf。

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 38.69%     | 39.14%     | 38.61%     |
| Fold 2   | 50.39%     | 42.12%     | 43.41%     |
| Fold 3   | 47.99%     | 43.40%     | 44.61%     |
| Fold 4   | 38.49%     | 37.75%     | 37.61%     |
| Fold 5   | 43.21%     | 39.13%     | 39.67%     |
| Fold 6   | 43.45%     | 39.70%     | 39.68%     |
| Fold 7   | 51.18%     | 39.26%     | 40.23%     |
| Fold 8   | 42.35%     | 38.32%     | 38.26%     |
| Fold 9   | 57.42%     | 44.60%     | 46.72%     |
| Fold 10  | 46.96%     | 40.81%     | 42.03%     |
| **Mean** | **46.01%** | **40.42%** | **41.09%** |

#### Train Result Table: 增加了nltk_bigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 71.16%     | 42.82%     | 44.67%     |
| Fold 2   | 37.32%     | 37.19%     | 36.80%     |
| Fold 3   | 45.94%     | 41.34%     | 42.12%     |
| Fold 4   | 52.75%     | 44.71%     | 45.99%     |
| Fold 5   | 48.41%     | 42.83%     | 44.10%     |
| Fold 6   | 43.15%     | 39.89%     | 40.45%     |
| Fold 7   | 49.92%     | 43.25%     | 44.42%     |
| Fold 8   | 46.46%     | 40.90%     | 41.36%     |
| Fold 9   | 47.06%     | 40.58%     | 41.54%     |
| Fold 10  | 36.83%     | 36.48%     | 36.03%     |
| **Mean** | **47.90%** | **41.00%** | **41.75%** |

#### Train Result Table: 增加了nltk_bigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 41.52%     | 40.95%     | 40.71%     |
| Fold 2   | 48.11%     | 42.43%     | 43.54%     |
| Fold 3   | 50.59%     | 44.60%     | 45.55%     |
| Fold 4   | 44.32%     | 42.81%     | 43.05%     |
| Fold 5   | 49.42%     | 40.71%     | 41.87%     |
| Fold 6   | 44.26%     | 39.23%     | 39.92%     |
| Fold 7   | 54.89%     | 45.69%     | 47.45%     |
| Fold 8   | 41.63%     | 40.48%     | 40.47%     |
| Fold 9   | 44.27%     | 39.76%     | 40.69%     |
| Fold 10  | 42.11%     | 37.69%     | 37.89%     |
| **Mean** | **46.11%** | **41.44%** | **42.11%** |

居然到了42.11%，太神奇了！	

#### Train Result Table: 增加了nltk_bigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 50.76%     | 41.16%     | 42.68%     |
| Fold 2   | 53.70%     | 44.18%     | 45.55%     |
| Fold 3   | 58.68%     | 41.17%     | 43.14%     |
| Fold 4   | 43.11%     | 38.80%     | 39.08%     |
| Fold 5   | 47.67%     | 42.70%     | 43.31%     |
| Fold 6   | 40.14%     | 37.65%     | 37.67%     |
| Fold 7   | 35.73%     | 35.14%     | 34.99%     |
| Fold 8   | 40.37%     | 39.16%     | 39.26%     |
| Fold 9   | 43.41%     | 39.00%     | 39.36%     |
| Fold 10  | 38.98%     | 38.29%     | 38.15%     |
| **Mean** | **45.25%** | **39.72%** | **40.32%** |

PS：感觉波动比较大

#### Train Result Table: 增加了nltk_bigram_t2 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 49.95%     | 43.17%     | 44.40%     |
| Fold 2   | 35.49%     | 37.19%     | 35.50%     |
| Fold 3   | 40.33%     | 37.95%     | 38.18%     |
| Fold 4   | 43.36%     | 42.00%     | 41.95%     |
| Fold 5   | 52.49%     | 38.65%     | 40.07%     |
| Fold 6   | 40.80%     | 38.09%     | 37.66%     |
| Fold 7   | 42.67%     | 39.64%     | 39.84%     |
| Fold 8   | 46.42%     | 41.70%     | 42.18%     |
| Fold 9   | 48.37%     | 43.45%     | 44.11%     |
| Fold 10  | 44.23%     | 43.26%     | 43.39%     |
| **Mean** | **44.41%** | **40.51%** | **40.73%** |


```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t2_with_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 46.05%     | 43.16%     | 43.66%     |
| Fold 2   | 44.76%     | 41.09%     | 41.70%     |
| Fold 3   | 42.40%     | 40.35%     | 40.27%     |
| Fold 4   | 54.25%     | 46.35%     | 47.36%     |
| Fold 5   | 46.90%     | 42.60%     | 43.50%     |
| Fold 6   | 53.48%     | 42.06%     | 43.60%     |
| Fold 7   | 39.37%     | 38.33%     | 38.28%     |
| Fold 8   | 44.81%     | 39.28%     | 39.82%     |
| Fold 9   | 37.47%     | 38.28%     | 37.30%     |
| Fold 10  | 47.81%     | 39.79%     | 40.81%     |
| **Mean** | **45.73%** | **41.13%** | **41.63%** |

#### Train Result Table: 增加了nltk_bigram_t2_with_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 41.71%     | 38.99%     | 39.12%     |
| Fold 2   | 43.03%     | 42.56%     | 42.51%     |
| Fold 3   | 44.69%     | 41.59%     | 42.09%     |
| Fold 4   | 43.64%     | 39.67%     | 40.13%     |
| Fold 5   | 46.81%     | 38.12%     | 38.39%     |
| Fold 6   | 49.46%     | 44.55%     | 45.83%     |
| Fold 7   | 48.18%     | 38.46%     | 39.39%     |
| Fold 8   | 49.81%     | 41.03%     | 42.02%     |
| Fold 9   | 49.54%     | 43.19%     | 44.24%     |
| Fold 10  | 42.86%     | 40.37%     | 40.61%     |
| **Mean** | **45.97%** | **40.85%** | **41.43%** |

#### Train Result Table: 增加了nltk_bigram_t2_with_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 56.37%     | 41.89%     | 43.74%     |
| Fold 2   | 36.19%     | 37.23%     | 36.51%     |
| Fold 3   | 43.45%     | 41.89%     | 41.94%     |
| Fold 4   | 39.66%     | 38.88%     | 38.52%     |
| Fold 5   | 43.75%     | 43.38%     | 43.16%     |
| Fold 6   | 46.65%     | 38.16%     | 38.17%     |
| Fold 7   | 39.78%     | 40.23%     | 39.77%     |
| Fold 8   | 48.65%     | 42.02%     | 43.30%     |
| Fold 9   | 54.07%     | 43.89%     | 46.35%     |
| Fold 10  | 55.75%     | 43.71%     | 46.11%     |
| **Mean** | **46.43%** | **41.13%** | **41.76%** |

#### 结论

nltk_bigram_t2 的效果比 t1 稍微好一些。41.40%以上

有 rf 的稍微稳定一些，没有rf的不太稳定。三次rf 平均值为 41.61%

### 计算 4

添加 nltk_bigram (freq 3), 分别测试 nltk_bigram / nltk_bigram_rf。

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t3 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 49.39%     | 40.26%     | 41.27%     |
| Fold 2   | 46.07%     | 38.21%     | 38.72%     |
| Fold 3   | 46.63%     | 39.35%     | 40.50%     |
| Fold 4   | 52.74%     | 44.73%     | 45.89%     |
| Fold 5   | 40.45%     | 40.61%     | 40.32%     |
| Fold 6   | 48.84%     | 40.77%     | 42.58%     |
| Fold 7   | 44.22%     | 39.93%     | 40.69%     |
| Fold 8   | 48.70%     | 43.36%     | 44.33%     |
| Fold 9   | 49.06%     | 39.53%     | 40.71%     |
| Fold 10  | 41.46%     | 40.96%     | 40.96%     |
| **Mean** | **46.76%** | **40.77%** | **41.60%** |

#### Train Result Table: 增加了nltk_bigram_t3 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 46.37%     | 39.10%     | 39.68%     |
| Fold 2   | 54.98%     | 48.93%     | 50.14%     |
| Fold 3   | 42.13%     | 38.21%     | 38.50%     |
| Fold 4   | 44.91%     | 40.69%     | 41.60%     |
| Fold 5   | 72.95%     | 45.30%     | 47.33%     |
| Fold 6   | 53.37%     | 41.14%     | 41.95%     |
| Fold 7   | 44.55%     | 38.83%     | 39.44%     |
| Fold 8   | 45.27%     | 38.14%     | 38.35%     |
| Fold 9   | 55.82%     | 43.31%     | 43.88%     |
| Fold 10  | 41.00%     | 37.90%     | 37.83%     |
| **Mean** | **50.13%** | **41.16%** | **41.87%** |

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t3_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 44.85%     | 40.34%     | 40.95%     |
| Fold 2   | 47.38%     | 41.81%     | 42.48%     |
| Fold 3   | 41.28%     | 37.79%     | 37.99%     |
| Fold 4   | 46.14%     | 41.13%     | 42.01%     |
| Fold 5   | 49.90%     | 43.00%     | 44.40%     |
| Fold 6   | 55.21%     | 42.26%     | 44.02%     |
| Fold 7   | 41.65%     | 40.94%     | 40.94%     |
| Fold 8   | 47.60%     | 43.53%     | 44.50%     |
| Fold 9   | 41.08%     | 38.58%     | 38.54%     |
| Fold 10  | 45.29%     | 40.38%     | 41.15%     |
| **Mean** | **46.04%** | **40.98%** | **41.70%** |

#### Train Result Table: 增加了nltk_bigram_t3_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 46.88%     | 39.80%     | 41.48%     |
| Fold 2   | 52.26%     | 45.11%     | 46.92%     |
| Fold 3   | 51.97%     | 42.45%     | 43.67%     |
| Fold 4   | 42.58%     | 40.51%     | 40.54%     |
| Fold 5   | 46.97%     | 42.28%     | 43.34%     |
| Fold 6   | 43.47%     | 41.55%     | 41.77%     |
| Fold 7   | 48.36%     | 40.29%     | 40.86%     |
| Fold 8   | 41.32%     | 40.63%     | 40.66%     |
| Fold 9   | 51.90%     | 39.52%     | 40.60%     |
| Fold 10  | 38.20%     | 39.40%     | 38.42%     |
| **Mean** | **46.39%** | **41.15%** | **41.83%** |

#### Train Result Table: 增加了nltk_bigram_t3_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 37.96%     | 37.25%     | 36.39%     |
| Fold 2   | 47.09%     | 40.57%     | 41.61%     |
| Fold 3   | 46.48%     | 39.90%     | 40.54%     |
| Fold 4   | 53.94%     | 44.83%     | 46.27%     |
| Fold 5   | 42.16%     | 42.24%     | 41.88%     |
| Fold 6   | 45.97%     | 39.14%     | 39.62%     |
| Fold 7   | 46.32%     | 42.95%     | 43.75%     |
| Fold 8   | 51.88%     | 42.41%     | 43.56%     |
| Fold 9   | 37.12%     | 37.31%     | 36.96%     |
| Fold 10  | 40.95%     | 39.63%     | 39.82%     |
| **Mean** | **44.99%** | **40.62%** | **41.04%** |

#### Train Result Table: 增加了nltk_bigram_t3_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 43.12%     | 38.98%     | 39.24%     |
| Fold 2   | 39.69%     | 38.91%     | 38.27%     |
| Fold 3   | 40.23%     | 39.35%     | 39.12%     |
| Fold 4   | 60.03%     | 45.42%     | 47.72%     |
| Fold 5   | 46.66%     | 42.02%     | 43.06%     |
| Fold 6   | 50.64%     | 44.53%     | 45.26%     |
| Fold 7   | 39.67%     | 39.72%     | 39.54%     |
| Fold 8   | 44.07%     | 39.72%     | 40.20%     |
| Fold 9   | 49.55%     | 42.29%     | 43.04%     |
| Fold 10  | 49.13%     | 41.12%     | 42.46%     |
| **Mean** | **46.28%** | **41.21%** | **41.79%** |

#### Train Result Table: 增加了nltk_bigram_t3_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 46.80%     | 41.89%     | 42.79%     |
| Fold 2   | 39.02%     | 38.03%     | 37.30%     |
| Fold 3   | 51.38%     | 47.05%     | 48.60%     |
| Fold 4   | 44.85%     | 39.73%     | 40.24%     |
| Fold 5   | 54.09%     | 42.33%     | 43.81%     |
| Fold 6   | 54.39%     | 44.80%     | 46.56%     |
| Fold 7   | 36.87%     | 36.36%     | 35.63%     |
| Fold 8   | 44.49%     | 39.60%     | 40.16%     |
| Fold 9   | 51.53%     | 44.29%     | 45.99%     |
| Fold 10  | 42.92%     | 40.12%     | 40.66%     |
| **Mean** | **46.63%** | **41.42%** | **42.17%** |

#### 结论

和 t2 没有很明显的差别。如果可以，之后不会使用 t1, t2 (因为容易过拟合问题。)

4次rf 平均值为 41.71%

### 计算 5

添加 nltk_bigram (freq 4), 分别测试 nltk_bigram / nltk_bigram_rf。

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t4 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 43.44%     | 38.52%     | 39.37%     |
| Fold 2   | 37.71%     | 37.93%     | 37.56%     |
| Fold 3   | 42.18%     | 40.07%     | 40.14%     |
| Fold 4   | 60.52%     | 48.23%     | 50.46%     |
| Fold 5   | 43.45%     | 40.80%     | 40.86%     |
| Fold 6   | 42.74%     | 40.46%     | 40.73%     |
| Fold 7   | 41.72%     | 36.17%     | 36.71%     |
| Fold 8   | 45.61%     | 40.72%     | 41.16%     |
| Fold 9   | 46.69%     | 39.11%     | 40.10%     |
| Fold 10  | 52.71%     | 46.03%     | 47.73%     |
| **Mean** | **45.68%** | **40.80%** | **41.48%** |

#### Train Result Table: 增加了nltk_bigram_t4 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 42.89%     | 41.38%     | 41.66%     |
| Fold 2   | 40.29%     | 38.59%     | 38.16%     |
| Fold 3   | 45.98%     | 42.01%     | 42.90%     |
| Fold 4   | 38.74%     | 38.72%     | 38.30%     |
| Fold 5   | 49.78%     | 39.11%     | 39.88%     |
| Fold 6   | 49.32%     | 41.05%     | 42.28%     |
| Fold 7   | 62.16%     | 44.98%     | 47.98%     |
| Fold 8   | 46.79%     | 41.73%     | 42.63%     |
| Fold 9   | 44.76%     | 39.53%     | 40.62%     |
| Fold 10  | 45.40%     | 39.88%     | 40.57%     |
| **Mean** | **46.61%** | **40.70%** | **41.50%** |

#### Train Result Table: 增加了nltk_bigram_t4 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 46.59%     | 41.70%     | 42.23%     |
| Fold 2   | 43.96%     | 38.32%     | 38.70%     |
| Fold 3   | 44.42%     | 37.58%     | 38.00%     |
| Fold 4   | 47.55%     | 40.71%     | 41.15%     |
| Fold 5   | 48.40%     | 44.10%     | 45.29%     |
| Fold 6   | 45.77%     | 40.54%     | 41.21%     |
| Fold 7   | 45.77%     | 36.80%     | 36.55%     |
| Fold 8   | 45.99%     | 41.31%     | 42.39%     |
| Fold 9   | 40.21%     | 40.26%     | 40.00%     |
| Fold 10  | 37.33%     | 35.22%     | 35.31%     |
| **Mean** | **44.60%** | **39.65%** | **40.08%** |

```
Using following features:
==============================
nltk_unigram_with_rf
nltk_bigram_with_rf
hashtag_with_rf
ners_existed
wv_google
wv_GloVe
sentilexi
emoticon
punction
elongated
==============================
```

#### Train Result Table: 增加了nltk_bigram_t4_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 43.12%     | 41.61%     | 41.82%     |
| Fold 2   | 60.35%     | 44.39%     | 46.62%     |
| Fold 3   | 37.14%     | 36.55%     | 36.33%     |
| Fold 4   | 54.18%     | 42.22%     | 44.47%     |
| Fold 5   | 53.24%     | 40.47%     | 41.18%     |
| Fold 6   | 48.84%     | 40.71%     | 41.93%     |
| Fold 7   | 39.02%     | 37.83%     | 37.88%     |
| Fold 8   | 40.54%     | 38.85%     | 38.87%     |
| Fold 9   | 40.44%     | 39.89%     | 39.58%     |
| Fold 10  | 48.16%     | 42.55%     | 43.46%     |
| **Mean** | **46.50%** | **40.51%** | **41.22%** |

#### Train Result Table: 增加了nltk_bigram_t4_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 45.19%     | 43.47%     | 43.72%     |
| Fold 2   | 44.13%     | 37.76%     | 38.28%     |
| Fold 3   | 39.98%     | 39.47%     | 39.19%     |
| Fold 4   | 38.19%     | 35.07%     | 35.24%     |
| Fold 5   | 40.23%     | 37.84%     | 38.12%     |
| Fold 6   | 50.83%     | 40.46%     | 42.10%     |
| Fold 7   | 40.33%     | 39.90%     | 39.52%     |
| Fold 8   | 51.75%     | 42.99%     | 44.59%     |
| Fold 9   | 44.65%     | 42.55%     | 42.80%     |
| Fold 10  | 50.51%     | 43.25%     | 44.89%     |
| **Mean** | **44.58%** | **40.28%** | **40.84%** |

#### Train Result Table: 增加了nltk_bigram_t4_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 47.30%     | 38.95%     | 39.22%     |
| Fold 2   | 35.40%     | 36.20%     | 35.42%     |
| Fold 3   | 47.53%     | 42.16%     | 43.19%     |
| Fold 4   | 45.60%     | 41.52%     | 42.04%     |
| Fold 5   | 52.20%     | 45.42%     | 47.19%     |
| Fold 6   | 45.19%     | 44.39%     | 44.46%     |
| Fold 7   | 41.74%     | 38.95%     | 39.44%     |
| Fold 8   | 42.73%     | 39.90%     | 40.51%     |
| Fold 9   | 38.45%     | 34.84%     | 35.04%     |
| Fold 10  | 53.24%     | 43.63%     | 44.48%     |
| **Mean** | **44.94%** | **40.60%** | **41.10%** |

结论：

t4的平均值为41%左右，没有t3效果好。

### 计算 6

添加 nltk_bigram (freq 5), 分别测试 nltk_bigram / nltk_bigram_rf。

#### Train Result Table: 增加了nltk_bigram_t5_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 36.44%     | 37.34%     | 36.81%     |
| Fold 2   | 46.90%     | 44.19%     | 44.95%     |
| Fold 3   | 44.04%     | 41.31%     | 41.74%     |
| Fold 4   | 41.91%     | 38.80%     | 39.11%     |
| Fold 5   | 36.49%     | 37.44%     | 36.28%     |
| Fold 6   | 46.92%     | 37.76%     | 38.80%     |
| Fold 7   | 36.54%     | 37.08%     | 36.38%     |
| Fold 8   | 53.30%     | 45.22%     | 46.70%     |
| Fold 9   | 40.52%     | 37.63%     | 37.45%     |
| Fold 10  | 54.61%     | 45.29%     | 47.39%     |
| **Mean** | **43.77%** | **40.21%** | **40.56%** |

#### Train Result Table: 增加了nltk_bigram_t5_rf 
| Fold     | Precision  | Recall     | F-1        |
| -------- | ---------- | ---------- | ---------- |
| Fold 1   | 44.62%     | 39.24%     | 39.80%     |
| Fold 2   | 45.56%     | 43.09%     | 43.65%     |
| Fold 3   | 37.23%     | 35.65%     | 35.52%     |
| Fold 4   | 51.52%     | 42.67%     | 43.71%     |
| Fold 5   | 50.35%     | 42.87%     | 44.03%     |
| Fold 6   | 38.48%     | 39.27%     | 38.49%     |
| Fold 7   | 44.97%     | 39.55%     | 39.82%     |
| Fold 8   | 41.63%     | 39.72%     | 39.93%     |
| Fold 9   | 44.54%     | 40.00%     | 41.02%     |
| Fold 10  | 37.89%     | 37.35%     | 37.02%     |
| **Mean** | **43.68%** | **39.94%** | **40.30%** |

#### 结论:

t5比不过t4。没有继续测试下的意义了。t3_rf是最好的。

## Dec. 6th 2017

This is the first time that I starting to make logs.

Until now, I have add these features to the SemEval2018_Task3 (abbr as S2018T3 in the future).

### Features

| Feature Name         | Description                              |
| -------------------- | ---------------------------------------- |
| nltk_unigram_with_rf | The unigram (each word) in each tweet. <br/>With a rf value weighting added. |
| hashtag_with_rf      | The hashtags (#morning, #like) int each tweet. <br/>With a rf value weighting added. |
| ners_existed         | Add ner information of each tweet.       |
| wv_google            | Word vector information of each tweet.   |
| wv_GloVe             | Word vector information of each tweet.   |
| sentilexi            | The dictionary of sentiment words in each tweet. |
| emoticon             | Each Emoticons in tweet, like :), `:).   |
| punction             | The punction of each tweet.              |
| elongated            | The elongated words, like !!!, hahaha, etc. |

### Efforts and results:

```
====================
Fold 1	: 41.68%
Fold 2	: 39.55%
Fold 3	: 35.18%
Fold 4	: 37.35%
Fold 5	: 36.81%
Fold 6	: 44.17%
Fold 7	: 41.97%
Fold 8	: 42.36%
Fold 9	: 37.36%
Fold 10	: 38.08%
====================
Mean	: 39.45%
```
The average of the F1-score is 39.45%-40%.


