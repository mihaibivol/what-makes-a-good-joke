# Classifiers

## Random forest

* read all jokes
* decide funny or not if log(score) < THRESHOLD
* use dictvectorizer + tfidf to extract features from the text
* use categories and subcategories as features
* put all data in random forest classifier
* predict funny or not on the rest of test data
* count good/bad predictions

To run, from root dir: `python -m classify.random_forest`

Sample output:

```
[Parallel(n_jobs=4)]: Done   1 out of  50 | elapsed:    0.6s remaining:   31.8s
[Parallel(n_jobs=4)]: Done  50 out of  50 | elapsed:   13.3s finished
[Parallel(n_jobs=4)]: Done   1 out of  50 | elapsed:    0.0s remaining:    0.2s
[Parallel(n_jobs=4)]: Done  50 out of  50 | elapsed:    0.0s finished
3132 good out of 4349. Bad: 1217.
72.02
```

Last line is good/total * 100.

Notes:

* tried ngram 2, 3 but score did not improve
* as THRESHOLD increases, accuracy increases (since there are more non funny than funny jokes)

## Multinomial Naive Bayes

* same routine for parsing data as Random Forest
* probability for word w to be in class C is calculated as such
![bayes](http://i.imgur.com/fRjvFlT.png)

(generated with http://www.sciweavers.org/free-online-latex-equation-editor)

To run, from root dir: `python -m classify.naive_bayes`

Sample output:
```
3209 good out of 4349. Bad: 1140.
73.79
```
