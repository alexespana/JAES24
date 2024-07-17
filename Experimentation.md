# Experimentation

In this section we will explain the experiments conducted to evaluate the effectiveness and performance of our approach compared to other experiments reported in the literature, such as the one presented in the paper [A Cost-efficient Approach to Building in Continuous Integration](https://ieeexplore.ieee.org/document/9284054). We will compare the results of our approach with the results of the experiment presented in the paper.


## Experiment 1
In this experiment, we will evaluate the effectiveness of our approach in terms of the number of captured failed builds by the classifier. To do this, we will follow these steps:
1. Collect all builds metadata from some examples repositories.
2. Extract the same features used in the [paper](https://ieeexplore.ieee.org/document/9284054) previously mentioned.
3. Train a random forest classifier using the train data (80%).
4. Predict outcomes of builds (success or failure) using the test data (20%).
5. Evaluate the classifier using the following metrics: accuracy, precision, recall, f1-score, confusion matrix and area under the ROC curve.
6. Repeat the experiment adding the features proposed in our approach.
7. Compare the results obtained in steps 4 and 5.

### Procedure
For this experiment, we will use the following repositories as test cases: [junit5](https://github.com/junit-team/junit5), [xls](https://github.com/google/xls) and [TypeScript](https://github.com/microsoft/TypeScript).

The feautures used in the paper are:
- **SC**: Number of changed lines.
- **FC**:Number of changed files.
- **TC**:Number of changed test lines.
- **NC**:Number of commits since the last build.

The features proposed in our approach are the previous ones plus:
- **FA**: Number of files added.
- **FM**: Number of files modified.
- **FR**: Number of files removed.
- **LA**: Number of lines added.
- **LR**: Number of lines removed.
- **UT**: Wheter tests have been written or not.

### Results

These metrics have been calculated considering that the true positive (TP) case is a failed build.

#### junit5: [https://github.com/junit-team/junit5](https://github.com/junit-team/junit5)

- Number of builds = `3850`
- Training size = `3080`
- Test size = `770`
- Ratio of pass/fail builds = `0.9690909090909091/0.03090909090909091`

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">junit5</th>
    <th>SmartBuildSkip</th>
    <th>Our approach</th>
    <th>Our approach without accumulation</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.966234</td>
    <td>0.962338</td>
    <td>0.966234</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.000000</td>
    <td>0.363636</td>
    <td>0.500000</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.000000</td>
    <td>0.153846</td>
    <td>0.076923</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.000000</td>
    <td>0.216216</td>
    <td>0.133333</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.456214</td>
    <td>0.980666</td>
    <td>0.982191</td>
  </tr>
</table>

- SmartBuildSkip: in this case, we see that the classifier has a fairly high accuracy, this may be simply because the number of failed builds is very low, so the classifier can predict the majority class with a high accuracy. The precision, recall and f1-score are `0`, which means that the classifier is not able to predict the failed builds. The AUC is `0.456214`, which means hat the model does not have almost discriminative capacity between positive and negative classes. 
- Our approach: we can see that the accuracy is the same as the SmartBuildSkip, but the precision, recall and f1-score are higher. We have a value of `0.363636` for precision, which is not a very good value, it indicates that `36.3636%` of the instances predicted as a failed build are actually failed builds. The recall is `0.153846`, which means that the `15.3846%` of the truly failed builds were correctly predicted by the model. The F1-score is `0.216216`, which is a low value, but it is higher than the SmartBuildSkip. The AUC is `0.980666`, which is a very high value, this means that the model is able to distinguish between the classes.

It is worth mentioning that these results have been obtained using a random forest classifier; however, using a decision tree classifier, the results are quite better (decision tree classifier):

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">junit5</th>
    <th>SmartBuildSkip</th>
    <th>Our approach</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.966234</td>
    <td>0.950649</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.000000</td>
    <td>0.396552</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.000000</td>
    <td>0.884615</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.000000</td>
    <td>0.547619</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.456214</td>
    <td>0.918786</td>
  </tr>
</table>

The value of the F1-score is higher than the random forest classifier, which means that the decision tree classifier is better at predicting the failed builds, although the precision and the recall are not a very good value.

#### xls: [https://github.com/google/xls](https://github.com/google/xls)

- Number of builds = `6341`
- Training size = `5072`
- Test size = `1269`
- Ratio of pass/fail builds = `0.9149976344425169/0.08500236555748304`


<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">xls</th>
    <th>SmartBuildSkip</th>
    <th>Our approach</th>
    <th>Our approach without accumulation</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.912530</td>
    <td>0.914106</td>
    <td>0.985028</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.000000</td>
    <td>0.505376</td>
    <td>0.942308</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.000000</td>
    <td>0.846847</td>
    <td>0.882883</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.000000</td>
    <td>0.632997</td>
    <td>0.911628</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.488653</td>
    <td>0.922568</td>
    <td>0.998320</td>
  </tr>
</table>

- SmartBuildSkip: as previously mentioned, the accuracy is high, but the precision, recall and f1-score are `0`, which means that the classifier is not able to predict the failed builds. The AUC is `0.488653`, which means that the model does not have almost discriminative capacity between positive and negative classes.
- Our approach: we have a higher accuracy than the SmartBuildSkip, and the precision, recall and f1-score are higher. The precision is `0.505376`, which is a very good value, it indicates that `50.5376%` of the instances predicted as a failed build are actually failed builds. The recall is `0.846847`, which means that the `84.6847%` of the truly failed builds were correctly predicted by the model. The F1-score is `0.632997`, which is a high value. The AUC is `0.922568`, which is a very high value, this means that the model is able to distinguish between the classes.

#### TypeScript: [https://github.com/microsoft/TypeScript](https://github.com/microsoft/TypeScript)

- Number of builds = `7177`
- Training size = `5741`
- Test size = `1436`
- Ratio of pass/fail builds = `0.9763132227950397/0.02368677720496029`

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">TypeScript</th>
    <th>SmartBuildSkip</th>
    <th>Our approach</th>
    <th>Our approach without accumulation</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.985376</td>
    <td>0.979109</td>
    <td>0.989554</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.000000</td>
    <td>0.390244</td>
    <td>0.650000</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.000000</td>
    <td>0.761905</td>
    <td>0.619048</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.000000</td>
    <td>0.516129</td>
    <td>0.634146</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.496113</td>
    <td>0.986185</td>
    <td>0.996298</td>
  </tr>
</table>

- SmartBuildSkip: the accuracy is high, but the precision, recall and f1-score are `0`, which means that the classifier is not able to predict the failed builds. The AUC is `0.496113`, which means that the model does not have almost discriminative capacity between positive and negative classes.
- Our approach: we have a higher accuracy than the SmartBuildSkip, and the precision, recall and f1-score are higher. The precision is `0.390244`, which is a good value, it indicates that `39.0244%` of the instances predicted as a failed build are actually failed builds. The recall is `0.761905`, which means that the `76.1905%` of the truly failed builds were correctly predicted by the model. The F1-score is `0.516129`, which is a high value. The AUC is `0.986185`, which is a very high value, this means that the model is able to distinguish between the classes.

As with the case of the xls repository, we have obtained better results using a `decision tree classifier`:

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">TypeScript</th>
    <th>SmartBuildSkip</th>
    <th>Our approach</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.985366</td>
    <td>0.980501</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.000000</td>
    <td>0.414634</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.000000</td>
    <td>0.809524</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.000000</td>
    <td>0.548387</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.496113</td>
    <td>0.896281</td>
  </tr>
</table>

The value of F1-score is much better than using the SmartBuildSkip or the random forest classifier.

---

To better compare the obtained results and understand the problem more comprehensively, we conducted the same experiment using like true positive (TP) the successful builds. The results are shown below for each repository:

#### junit5

- Number of builds = `3850`
- Training size = `3080`
- Test size = `770`
- Ratio of pass/fail builds = `0.9690909090909091/0.03090909090909091`

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">junit5</th>
    <th>TP = failed build</th>
    <th>TP = successful build</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.966234</td>
    <td>0.966234</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.500000</td>
    <td>0.968668</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.076923</td>
    <td>0.997312</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.133333</td>
    <td>0.982781</td>
  </tr>
    <td><b>AUC</b></td>
    <td>0.982191</td>
    <td>0.982191</td>
  </tr>
</table>

#### xls

- Number of builds = `6341`
- Training size = `5073`
- Test size = `1268`
- Ratio of pass/fail builds = `0.9149976344425169/0.08500236555748304`

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">xls</th>
    <th>TP = failed build</th>
    <th>TP = successful build</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.987382</td>
    <td>0.987382</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.970297</td>
    <td>0.988860</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.882883</td>
    <td>0.997407</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.924528</td>
    <td>0.993115</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.998661</td>
    <td>0.998661</td>
  </tr>
</table>

### TypeScript

- Number of builds = `7177`
- Training size = `5742`
- Test size = `1435`
- Ratio of pass/fail builds = `0.9763132227950397/0.02368677720496029`

<table align="center" border="1">
  <tr>
    <th style="border-width: 3px;">TypeScript</th>
    <th>TP = failed build</th>
    <th>TP = successful build</th>
  </tr>
  <tr>
    <td><b>ACC</b></td>
    <td>0.988153</td>
    <td>0.988153</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.583333</td>
    <td>0.995039</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.666667</td>
    <td>0.992928</td>
  </tr>
  <tr>
    <td><b>F1-score</b></td>
    <td>0.622222</td>
    <td>0.993982</td>
  </tr>
  <tr>
    <td><b>AUC</b></td>
    <td>0.994814</td>
    <td>0.994814</td>
  </tr>
</table>
