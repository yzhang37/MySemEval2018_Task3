# MySemEval2018_Task3 Experimental Logs

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