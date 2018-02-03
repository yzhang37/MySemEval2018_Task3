# 第三阶段记录文件

2018-01-22

现在到了比赛的第二阶段 (TaskB)，之前的 TaskA 还是有诸多的问题

* 二分类问题的错误

  -[x] 由于之前的错误，修改 `Trainer.py` 文件如下：对于测试的情况，使用所有的训练数据，去评测测试结果。

* 二分类问题的增强：

  1. 如何在多次交叉验证的情况下，使用交叉验证的模型缓存来评测结果？(没有必要这么做，因为平时用的是多折交叉，而现在是所有的训练数据，因此平时的数据缓存了也没有用）。
  2. 训练模型文件过多，导致的混乱问题。

* 四分类问题

  -[x] 添加 url 特征


  -[ ] ensemble (投票处理)：
       -[ ] 10折，(使用 Hill Climbing)挑选特征和算法
       -[ ] 训练模型，然后测试，得到结果1（for emergency）

* 四个二分类问题

  -[ ] 0 | 1, 2, 3， 挑选特征(1)，挑选算法，然后**取概率**
  -[ ] 1 | 0, 2, 3
  -[ ] 2 | 0, 1, 3 （二分类的数据太少，要翻十倍）
  -[ ] 3 | 0, 1, 2 （三分类的数据太少，要翻十倍）


## 2018-01-25 记录

添加了 url 特征，并且使用了一些函数进行测试，结果大致如下:

（由于本次实验的四分类使用 F1-macro 方式进行计算）

### 使用 Liblinear SVM

| 添加项目               | F1-marco                                 | 再次运行结果如下   |
| ------------------ | ---------------------------------------- | ---------- |
| 不使用 Url Unigram    | $\frac{0.712455+0.649360+0.254167+0.066390}{4}=0.420593$ | $0.432609$ |
| Url Unigram **T1** | $\frac{0.710811+0.649750+0.282700+0.074074}{4}=0.429334$ | $0.421364$ |
| Url Unigram **T2** | $\frac{0.712627+0.655994+0.280922+0.105691}{4}=0.438809$ | $0.433040$ |
| Url Unigram **T3** | $\frac{0.703121+0.648496+0.241015+0.067511}{4}=0.415036$ | $0.430181$ |
| Url Unigram **T4** | $\frac{0.703121+0.650534+0.280922+0.074689}{4}=0.427334$ | $0.424290$ |
| Url Unigram **T5** | $\frac{0.709412+0.649573+0.278119+0.075630}{4}=0.428183$ | $0.426313$ |

结果：增加了 Url Unigram 后，增加的效果不是非常显著。

## 2018-01-26 记录

增加4个2分类的使用。



在原基础上，修改 main 的运行过程，改为四组二分类的程序。

增加要求：

-[x] 每个 ensemble 文件都只有0 和 1两个分类


-[x] 每个 ensemble 必须使用概率而不是简单一个分类


-[x] 对于每个二分类来说，都和自己所对应的这个 Golden Label 进行比较，然后进行 ensemble


## 2018-01-27

今日四件事情

-[x] 基础的跑出结果


-[x] 修改为概率显示。


-[x] 增加2，3分类的数据量（10倍）


-[ ] 使用 ensemble


## 2018-01-28

以往进行四分类算法分析的时候，最大的影响就是受到了2、3两类的影响，因为这两类的F1值太低的缘故。现在在执行4个类的二分类算法的过程中，发现2分类和3分类的数据量太少，相比其他分类数据量太少。因此使用数据量翻倍的方式进行测试。

对于二分类，1为原始分类的2，0为原始分类的其他类，使用 Scikit Learn Logistic Regression，效果如下：

### 数据量不翻倍的情况

**Evaluation on Scikit-Learn LogisticRegression**
row = predicted, column = truth
```
  0      1     
0 3459.0 269.0 
1 59.0   47.0  
```

0. precision 0.927843 recall 0.983229	 F1 0.954734
1. precision 0.443396 recall 0.148734.  F1 0.222749


* Overall accuracy rate = 0.914450
* Average precision 0.685620  recall 0.565982 F1 0.588741


### 二分类翻6倍，三分类翻10的效果

```
row = predicted, column = truth
  0      1     
0 3387.0 242.0 
1 131.0  74.0  

0 	precision 0.933315 	recall 0.962763	 F1 0.947810
1 	precision 0.360976 	recall 0.234177	 F1 0.284069
* Overall accuracy rate = 0.902713
* Average precision 0.647145 	 recall 0.598470	 F1 0.615940
```

### 二分类和三分类都翻10倍的效果

```
Evaluation on Scikit-Learn LogisticRegression
------------------------------------------------------------
row = predicted, column = truth
  0      1     
0 3358.0 230.0 
1 160.0  86.0  

0 	precision 0.935897 	recall 0.954520	 F1 0.945117
1 	precision 0.349593 	recall 0.272152	 F1 0.306050
* Overall accuracy rate = 0.898279
* Average precision 0.642745 	 recall 0.613336	 F1 0.625583
------------------------------------------------------------
```

可以看出来很明显，1分类的正确率变高了。



### 今天任务：

-[ ] 有一些算法是不能显示一个分类的概率的，将这种算法可剔除（不能用在当前的多个二分类的方式进行计算）


-[ ] 执行自动ensemble算法，评估不同的乘法比例下，效果最好。

| 算法                                       | 支持    | 注释                                       |
| ---------------------------------------- | ----- | ---------------------------------------- |
| LibLinear SVM                            | True  |                                          |
| Scikit-Learn AdaBoostClassifier          | True  |                                          |
| Scikit-Learn DecisionTree                | False | 评测的概率只有 1.0 和 0.0 两个分类，无法进行比较            |
| Scikit-Learn KNN Classifier              | False | 存在概率只有 1.0 和 0.0 的                       |
| Scikit-Learn LogisticRegression          | True  |                                          |
| Scikit-Learn NaïveBayes                  | False | 评测的概率只有 1.0 和 0.0 两个分类，用的是高斯NB           |
| Scikit-Learn RandomForestClassifier      | True  |                                          |
| Scikit-Learn Stochastic Gradient Descent (hinge) | False | probability estimates are not available for loss='hinge'<br/>loss修改为 log 或 modified_huber 就可以进行评测了。尝试一下效果, log 比 modified_huber 效果更加好，因此使用 log. |
| Scikit-Learn SVM                         | False | 不支持                                      |
| Scikit-Learn VotingClassifier            | False | predict_proba is not available when voting='hard' |
|                                          |       |                                          |

因为时间原因，放弃使用部分算法进行4分类训练，只使用如下的算法：

LibLinear SVM, AdaBoost, LogReg, RandomForest和SGDC(log)

## 2018-01-29

针对原始的数据，2分类扩大至6倍，3分类扩大至10倍，保证0，1，2，3分类的数据都在1800左右差不多

使用 LibLinear SVM, AdaBoost, LogReg, RandomForest和SGDC(log) 的训练结果，分别进行合并得到如下结果:

```
Using Scikit-Learn LogisticRegression for label '0', Scikit-Learn LogisticRegression for label '1', Scikit-Learn LogisticRegression for label '2', LibLinear SVM for label '3'

row = predicted, column = truth
  0      1     2     3     
0 1356.0 412.0 133.0 123.0 
1 354.0  866.0 63.0  36.0  
2 114.0  69.0  103.0 14.0  
3 99.0   43.0  17.0  32.0  

0 	precision 0.669960 	recall 0.705148	 F1 0.687104
1 	precision 0.656558 	recall 0.623022	 F1 0.639350
2 	precision 0.343333 	recall 0.325949	 F1 0.334416
3 	precision 0.167539 	recall 0.156098	 F1 0.161616
* Overall accuracy rate = 0.614763
* Average precision 0.459348 	 recall 0.452554	 F1 0.455622
```

因此，使用训练数据，将2和3分类进行翻倍，2分类扩大至6倍，3分类扩大至10倍，然后分别训练

Using Scikit-Learn LogisticRegression for label '0', Scikit-Learn LogisticRegression for label '1', Scikit-Learn LogisticRegression for label '2', LibLinear SVM for label '3'.

