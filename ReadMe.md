
Paraphrase identification in Python 3 using scikit-learn
===

A solution to the [Semeval Paraphrase identification task](http://alt.qcri.org/semeval2015/task1/)

It's a reimplementation and extension of [ASOBEK](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval011.pdf).

Currently only adds as a feature, the dot product of the sums of [word2vec](https://code.google.com/p/word2vec/) vectors
from both tweets and replaces SVC with AdaBoost-ed decision trees.  

Currently the performance of the method is unstable, sometimes yielding an F1
score of 0.6903 (beating ASOBEK's 0.674) and sometimes as low as 0.63.

The word2vec database is  "pre-trained vectors trained on part of Google News"
from the word2vec website.

#### Documented by Zihan Li:

Getting start:

Download Google news pre-trained Google News corpus from [https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) 

```shell
git clone https://github.com/lizihan021/paraphrase
cd paraphrase
# put GoogleNews-vectors-negative300.bin.gz at root directory
python3 -m pyword2vec.__main__ 
```

This takes 30 min on 2.7GHz i7. Be patient. Then you will get `filemap.w2v`  at root directory.

Then you need to get `train.data,test.data,test.label` from SemEval-2015 Task 1 organizers. 

I got a sample data and test cases of 100 rows, you can use them to run the program, but the result should be incorrect since train.data and test.data are basically the same file.



