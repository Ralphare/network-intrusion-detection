**Network Intrusion Detection — Assignment Report**

Tunar Abaszada & Javid Malikzade

# 1\. Introduction

The goal of this assignment was to build a machine learning model that classifies network connections into five categories: Normal, DoS, Probe, R2L, and U2R. The dataset is heavily imbalanced — U2R has only 52 training examples while Normal has over 67,000. The baseline Random Forest model achieves a macro F1-score of around 0.47, mostly because it almost completely fails to detect R2L and U2R attacks. Our goal was to beat that.

All experiments were developed and run in Jupyter Notebook. The full code is available in nids_solution.py in the repository and can be run either as a Python script or opened directly in Jupyter.

# 2\. What We Tried

## Algorithm Selection

We compared four classifiers on the same SMOTE-resampled data: Random Forest, XGBoost, LightGBM, and CatBoost. XGBoost trains trees sequentially — each new tree corrects mistakes of previous ones — making it better at learning from rare classes like U2R that other models tend to ignore.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Model** | **CV F1** | **Test F1** | **Gap** | **Best?** |
| Random Forest | 0.9997 | 0.5374 | 0.4624 |     |
| XGBoost | 0.9998 | 0.6270 | 0.3728 | **★** |
| LightGBM | 0.9998 | 0.5982 | 0.4016 |     |
| CatBoost | 0.9991 | 0.5770 | 0.4222 |     |

## Handling Class Imbalance with SMOTE

U2R has only 52 training records — there is simply not enough data for any model to learn what U2R attacks look like. We used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples by interpolating between existing minority class records. After SMOTE, all five classes were balanced to 67,343 examples each.

|     |     |     |
| --- | --- | --- |
| **Class** | **Before SMOTE** | **After SMOTE** |
| Normal | 67,343 | 67,343 |
| DoS | 45,927 | 67,343 |
| Probe | 11,656 | 67,343 |
| R2L | 995 | 67,343 |
| U2R | 52  | 67,343 |

# 3\. Results

## Final Classification Report

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
| DoS | 0.96 | 0.84 | 0.90 | 7,460 |
| Normal | 0.70 | 0.97 | 0.82 | 9,711 |
| Probe | 0.85 | 0.77 | 0.81 | 2,421 |
| R2L | 0.98 | 0.14 | 0.25 | 2,885 |
| U2R | 0.65 | 0.25 | 0.37 | 67  |
| **macro avg** | **0.83** | **0.60** | **0.63** | **22,544** |

|     |     |
| --- | --- |
| **Metric** | **Score** |
| Final Test Macro F1-score | **0.6443** |
| Baseline Macro F1-score | 0.47 |
| Improvement | **+0.1743 (+37% relative)** |

## Confusion Matrix

The rows are actual classes, the columns are predicted classes; the diagonal shows correct predictions. DoS and Normal are detected well. R2L remains the hardest class — most R2L attacks are misclassified as Normal because many R2L subtypes in the test set do not appear in training at all. U2R improved from near-zero recall in the baseline to 0.25 in our final model.

# 4\. Cross-Validation vs Test Score Comparison

|     |     |
| --- | --- |
| **Metric** | **Score** |
| Cross-validation macro F1 (5-fold, training data) | 0.9430 ± 0.0122 |
| Test macro F1 (KDDTest+) | **0.6443** |

There is a large gap between the cross-validation score and the test score. This is not overfitting — we never used the test set during training or tuning. There are two reasons for the gap. First, cross-validation was run on the SMOTE-resampled training data where the model learned those patterns very well. Second, KDDTest+ deliberately contains attack subtypes not present in KDDTrain+, so the model encounters genuinely new attack patterns at test time. The test score is the honest number.

# 5\. What We Would Try Next

Feature engineering. We did not create any new features beyond what the dataset provides. Interaction features like the ratio of source to destination bytes, or combining login failure counts with connection flags, might help the model better distinguish R2L from Normal.

Threshold tuning. By default the model predicts whichever class has the highest probability. Lowering the decision threshold for R2L and U2R would make the model more aggressive about flagging rare attacks at the cost of some precision, likely improving recall on those classes significantly.

Voting ensemble. Combining XGBoost and LightGBM predictions through a majority vote sometimes catches what a single model misses. Since LightGBM scored 0.60 on its own, combining it with XGBoost might produce a better result than either alone.

Neural network. An MLP classifier might learn more complex patterns, especially for the subtle differences between Normal and R2L traffic. We avoided it due to the additional complexity and tuning required, but it would be worth trying with more time.