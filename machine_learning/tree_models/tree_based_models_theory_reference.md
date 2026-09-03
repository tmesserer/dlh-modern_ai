# Tree-Based Models — Theoretical Reference

> Companion notes for `machine_learning/tree_models` (12 tasks, Wine dataset). Organized around the project's Learning Objectives — if you can explain every numbered section below in your own words, you've met the bar the project sets ("without the help of Google").

---

## 1. What is a decision tree classifier?

A decision tree is a **recursive, greedy partitioning** of the feature space. Every internal node asks a single yes/no question about one feature ("is `proline` ≤ 755?"), every edge is an answer, and every leaf is a class prediction. Scikit-learn implements a version of **CART** (Classification And Regression Trees).

Why it matters conceptually:
- It's **non-parametric** — no assumption about the underlying data distribution.
- Decision regions are **axis-aligned rectangles** in feature space (contrast with linear models, which cut with a single hyperplane).
- It naturally captures **feature interactions** and non-linear boundaries without you engineering them.
- It's **greedy**: at each node it picks the locally best split, not the globally best tree. Finding the globally optimal tree is NP-hard, so greedy construction is the practical compromise.

## 2. How do decision trees make splits?

At each node, the algorithm scans every feature and every candidate threshold, and picks the split that most reduces **impurity** in the resulting children.

**Gini impurity** (sklearn's default, `criterion='gini'`):
```
Gini(node) = 1 − Σ pᵢ²
```
where `pᵢ` is the proportion of class *i* samples in that node. It's the probability of misclassifying a randomly picked sample if you labeled it according to the node's class distribution. Gini = 0 means the node is pure.

**Entropy** (`criterion='entropy'`), from information theory:
```
Entropy(node) = − Σ pᵢ · log2(pᵢ)
```
Splits are chosen by maximizing **information gain** = impurity(parent) − weighted-average impurity(children).

Gini vs. entropy in practice: they usually agree on the "obviously good" splits. Gini is marginally cheaper (no logarithm) and is what sklearn defaults to; entropy is marginally more sensitive to shifts in class distribution. This is a minor knob, not a make-or-break decision.

## 3. What is pre-pruning vs. post-pruning?

An unconstrained tree grows until leaves are pure (or another stopping rule hits) — which usually means it memorizes noise in the training set: low bias, very high variance. Pruning is how you trade a little bias for a lot less variance.

- **Pre-pruning (early stopping):** constrain growth *while building* the tree — `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, `max_leaf_nodes`. Fast, prevents overfitting proactively.
  - Risk: the **horizon effect** — a split that looks weak right now might unlock a great split one level deeper, and pre-pruning can stop before it gets the chance.
  - How you tune it: search over the hyperparameter grid with cross-validation (`GridSearchCV` — see §7).
- **Post-pruning:** grow the *full* tree first, then cut back branches that don't earn their complexity. Scikit-learn's method is **cost-complexity pruning** (§4).

Rule of thumb: pre-pruning is cheaper and simpler; post-pruning is more thorough because it lets the tree "see" the whole picture before deciding what to discard.

## 4. What is `ccp_alpha` in pruning?

Cost-complexity pruning defines, for any subtree `T`, a regularized cost:
```
R_α(T) = R(T) + α · |T|
```
- `R(T)` = total impurity summed over the leaves of `T`
- `|T|` = number of leaves (terminal nodes) — the complexity penalty
- `α ≥ 0` (`ccp_alpha`) = how heavily you penalize complexity

At `α = 0`, there's no penalty and the fully grown tree wins. As `α` increases, the optimal `R_α`-minimizing subtree gets smaller — whole branches ("weakest links," the ones whose removal increases `R(T)` the least per leaf lost) get collapsed.

`cost_complexity_pruning_path` doesn't just give you one tree — it gives you the entire **nested sequence** of optimal subtrees `T₀ ⊃ T₁ ⊃ ... ⊃ {root}`, each one optimal over a range of `α`, plus the corresponding `ccp_alphas` and `impurities` arrays. That's the object you sweep over in the pruning tasks.

**What you'll see when you plot it:** training accuracy decreases monotonically as `α` grows (less capacity to memorize). Test accuracy usually rises first (less overfitting) then falls (underfitting) — a textbook bias-variance curve you can point to directly on your own plot. Picking the "best" `α` is model selection: among the candidates tied on test accuracy, prefer the smallest train/test gap (better generalization), and among further ties, prefer the *largest* `α` (simpler, more regularized tree).

## 5. Why combine trees at all? (Ensemble learning, general)

A single tree is low-bias/high-variance — small changes in training data can produce a very different tree. Ensembles combine many base learners so that individual errors average out. There are two dominant strategies, distinguished by **how the base models are trained and combined**:

| | Bagging | Boosting |
|---|---|---|
| Training scheme | Parallel, independent | Sequential, each model corrects the last |
| Data used per model | Bootstrap resample of rows | Full data, reweighted / residual-fit each round |
| Combination rule | Average / majority vote | Weighted vote or weighted sum |
| Primarily reduces | Variance | Bias (and often variance too, if regularized) |
| Overfitting risk | Low — more trees rarely hurts | Higher — too many rounds *can* overfit |
| Example algorithms | Random Forest | AdaBoost, Gradient Boosting, XGBoost, LightGBM |

(A third paradigm, **stacking**, trains a meta-model on top of diverse base models' outputs — not used in this project, but worth knowing it exists as the third leg of the ensemble triangle.)

## 6. How does a random forest improve over a single tree?

Random Forest = bagging + one extra trick, and both pieces matter:

1. **Bootstrap sampling (bagging):** each tree trains on a random resample of the training rows (with replacement). On average, each tree only sees ~63.2% of the unique rows — the rest ("out-of-bag," OOB) can serve as a free validation set for that tree (`oob_score=True`).
2. **Random feature subsetting at each split:** at every node, only a random subset of features is considered (`max_features`, default `'sqrt'` for classification) — *this* is the part that makes it a "random" forest rather than just "bagged trees."

Why both matter together: bagging alone still produces trees that are fairly *correlated* with each other (they tend to split on the same dominant features first). Randomly restricting the feature subset at each split **decorrelates** the trees. This matters because the variance of an average of `n` correlated variables doesn't shrink like `1/n` — it's bounded below by their average pairwise correlation. Less correlation → more effective variance reduction from averaging → a forest that generalizes better than any single member tree, without needing to prune individual trees nearly as aggressively.

Prediction: classification is decided by majority vote (or averaged class probabilities) across all trees.

## 7. What is feature importance in random forests?

The standard sklearn metric is **Mean Decrease in Impurity (MDI)**, a.k.a. Gini importance: for each feature, sum the (sample-weighted) impurity decrease it produced at every split where it was used, across every tree, then normalize so all importances sum to 1. Exposed as `.feature_importances_` on a trained model.

Caveat worth knowing (not required by the tasks, but genuinely useful): MDI is **biased toward high-cardinality / continuous features** — they simply get more opportunities to be chosen as a "good enough" split — and it's computed on training data, so it can inflate features that are really just good for overfitting. **Permutation importance** (shuffle one feature's values and measure the drop in test-set score) is a more robust, model-agnostic alternative if you ever need it beyond this project.

## 8. What is boosting?

Boosting builds an **additive model** sequentially:
```
F(x) = Σₘ learning_rate · fₘ(x)
```
where each `fₘ` is a weak learner (commonly a shallow tree, sometimes literally a 1-split "stump"). Each new learner is trained specifically to fix what the ensemble has gotten wrong *so far*. This turns many "slightly better than random" learners into one strong learner.

## 9. How does AdaBoost differ from Gradient Boosting?

**AdaBoost (Adaptive Boosting):**
- Maintains a weight on every training sample, uniform at the start.
- After each weak learner trains, samples it misclassified get **higher weight** (so the next learner focuses on the hard cases), correctly classified samples get lower weight.
- Each weak learner also earns a **vote weight** based on its own accuracy — more accurate learners get more say.
- Final prediction = weighted majority vote across all weak learners.
- Weakness: sensitive to noisy data/outliers, since they keep getting up-weighted round after round.

**Gradient Boosting:**
- More general framework. Instead of reweighting samples, each new tree is fit to the **negative gradient of the loss function** with respect to the current ensemble's predictions — for squared-error loss this is literally the residuals; for log-loss (classification) it's the "pseudo-residuals."
- This is gradient descent performed **in function space**, where each descent step is approximated by fitting a small tree.
- `learning_rate` shrinks each new tree's contribution (shrinkage regularization) — smaller steps + more trees generally generalizes better than fewer, larger steps.
- More flexible than AdaBoost (works with any differentiable loss), typically stronger, but more hyperparameters and easier to overfit if unregularized.

**One-line answer:** AdaBoost corrects mistakes by *reweighting examples*; Gradient Boosting corrects mistakes by *fitting the residual error directly* via gradient descent.

## 10. What are XGBoost and LightGBM?

Both are production-grade, heavily optimized implementations of the gradient boosting idea — not a different algorithm family, but engineering (and some mathematical) refinements on top of §9's Gradient Boosting.

**XGBoost (Extreme Gradient Boosting):**
- Uses a **second-order Taylor expansion** of the loss (gradient *and* Hessian) to choose splits more precisely and converge faster than first-order gradient boosting.
- Adds **explicit L1/L2 regularization** on leaf weights directly in the objective — regularization is built into the loss, not just imposed via `learning_rate`/depth.
- Handles **missing values natively** (learns a default split direction per node).
- Highly parallelized, cache-aware split-finding; can operate out-of-core for huge datasets.

**LightGBM (Light Gradient Boosting Machine):**
- **Leaf-wise (best-first) tree growth** instead of level-wise: at each step it splits whichever leaf gives the largest loss reduction, rather than expanding every leaf at the current depth. Produces deeper, asymmetric trees — often more accurate, but more prone to overfitting on small datasets (control with `num_leaves` / `max_depth`).
- **Histogram-based binning** of continuous features by default → much faster split-finding, lower memory.
- **GOSS** (Gradient-based One-Side Sampling): keeps the samples with large gradients (the ones the model is still getting wrong) and randomly subsamples the rest — trains faster with minimal accuracy loss.
- **EFB** (Exclusive Feature Bundling): bundles mutually-exclusive sparse features together, cutting effective dimensionality — a big win on sparse/high-cardinality data.
- Generally the fastest to train, especially as data scales up. On a small dataset like Wine, the speed gap you measure in Task 11 will be much less dramatic than it would be on a dataset with millions of rows — worth noting in your writeup if your timing results look "close."

## 11. Bagging vs. boosting — the direct answer

Bagging trains independent models **in parallel** on bootstrap resamples of the data and averages them — this primarily reduces **variance**, and adding more trees essentially never hurts. Boosting trains models **sequentially**, each one correcting the errors of the ensemble so far, combined via a weighted vote/sum — this primarily reduces **bias**, but can increase variance/overfit if you don't control `n_estimators` and `learning_rate`.

## 12. How do you evaluate classifier performance?

Per class (treated one-vs-rest for multi-class problems):
- **Precision** = TP / (TP + FP) — "of everything I predicted as class X, how much actually was?" Cost of false positives.
- **Recall** = TP / (TP + FN) — "of everything that is class X, how much did I catch?" Cost of false negatives.
- **F1-score** = harmonic mean of precision and recall = `2PR / (P + R)` — penalizes an imbalance between precision and recall more than a plain average would.
- **Support** = number of true instances of that class in `y_true`.

Aggregation across classes:
- **Macro avg** — unweighted mean across classes. Every class counts equally, so it's sensitive to poor performance on a small/rare class.
- **Weighted avg** — mean weighted by each class's support. Reflects the dataset's actual composition; dominated by majority classes.
- **Accuracy** — overall fraction correct. Can be misleading under class imbalance, which is exactly why the per-class table matters more than a single accuracy number.

`sklearn.metrics.classification_report(y_true, y_pred, target_names=...)` builds this whole table as a formatted string (or a dict, with `output_dict=True`).

## 13. Hyperparameter search: `GridSearchCV`

Exhaustively evaluates every combination in a parameter grid using **k-fold cross-validation** (default `cv=5`): split the training data into k folds, train on k−1, validate on the held-out fold, rotate through all folds, average the score. This gives a much more robust generalization estimate than a single train/validation split, at the cost of k× the training time. Exposes `.best_params_`, `.best_estimator_`, `.best_score_` after fitting.

Why grid search jointly rather than tuning one hyperparameter at a time: `max_depth`, `min_samples_leaf`, and `min_samples_split` **interact** — the best `max_depth` depends on what `min_samples_leaf` is set to, and vice versa. A joint grid search over the combination avoids missing good combinations that a sequential one-at-a-time search would.

---

## Knowledge catalog — API reference

| Name | Module | What it does | Params/attributes you'll touch |
|---|---|---|---|
| `DecisionTreeClassifier` | `sklearn.tree` | CART classifier | `criterion`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha`, `random_state` |
| `.fit(X, y)` | any sklearn estimator | Trains the model in place | — |
| `.predict(X)` | any sklearn estimator | Returns predicted labels | — |
| `.get_params()` | any sklearn estimator | Returns the estimator's hyperparameter dict | — |
| `.cost_complexity_pruning_path(X, y)` | `DecisionTreeClassifier` method | Returns `ccp_alphas` and `impurities` for post-pruning | — |
| `tree.export_text(clf, feature_names=...)` | `sklearn.tree` | Text representation of a trained tree's decision rules | — |
| `tree.plot_tree(clf, ...)` | `sklearn.tree` | Graphical rendering of a trained tree | `feature_names`, `class_names`, `filled` |
| `classification_report(y_true, y_pred, target_names=...)` | `sklearn.metrics` | Precision/recall/F1/support table | `output_dict` |
| `GridSearchCV(estimator, param_grid, cv=...)` | `sklearn.model_selection` | Exhaustive hyperparameter search with cross-validation | `param_grid`, `cv`, `scoring` |
| `RandomForestClassifier` | `sklearn.ensemble` | Bagged, decorrelated ensemble of trees | `n_estimators`, `max_features`, `random_state` |
| `.feature_importances_` | `RandomForestClassifier` / `DecisionTreeClassifier` attribute | Per-feature Gini-importance array | — |
| `AdaBoostClassifier` | `sklearn.ensemble` | Adaptive boosting ensemble | `n_estimators`, `random_state` |
| `GradientBoostingClassifier` | `sklearn.ensemble` | Sequential gradient boosting | `n_estimators`, `learning_rate`, `random_state` |
| `XGBClassifier` | `xgboost` | Regularized, second-order gradient boosting | `n_estimators`, `random_state` |
| `LGBMClassifier` | `lightgbm` | Fast, leaf-wise gradient boosting | `n_estimators`, `random_state`, `verbose` |

---

## Mapping the 12 tasks to the theory above

| Task | Title | Concept it's testing | Section |
|---|---|---|---|
| 0 | Decision Tree Classifier | Building an unconstrained CART classifier | §1 |
| 1 | Train a Tree-Based Classifier | `.fit()` mechanics — generic across every model in this project | catalog |
| 2 | View the Decision Rules | Reading a trained tree's split logic as text | §2, catalog (`export_text`) |
| 3 | Generate Predictions | `.predict()` | catalog |
| 4 | Evaluate Classifier Performance | Precision / recall / F1 | §12 |
| 5 | Pre-Pruning | `GridSearchCV` over `max_depth`, `min_samples_leaf`, `min_samples_split` | §3, §13 |
| 6 | Retrieve Pruning Path | `cost_complexity_pruning_path`, `ccp_alphas` | §4 |
| 7 | Train & Evaluate Across `ccp_alphas` | The bias-variance curve, made visible | §4 |
| 8 | Best `ccp_alpha` | Model selection / tie-breaking logic | §4 |
| 9 | Random Forest Classifier | Bagging + random feature subsetting | §6 |
| 10 | Feature Importance | Gini importance ranking | §7 |
| 11 | Boosting | AdaBoost vs. Gradient Boosting vs. XGBoost vs. LightGBM | §8, §9, §10 |

---

## Self-check before you start coding

Try to answer each of these out loud, from memory, before opening a task file:
1. What does Gini impurity measure, and why does a lower value mean a "purer" node?
2. Why does an unconstrained tree tend to overfit?
3. What's the actual mathematical object `ccp_alpha` penalizes?
4. Why does randomly restricting features at each split matter *in addition to* bagging rows?
5. In one sentence: what does a gradient boosting round actually fit a new tree to?
6. What specifically does XGBoost add on top of plain gradient boosting? What does LightGBM add?
7. When would weighted avg and macro avg in a classification report diverge a lot — and why?

If any of these feel shaky, that's the section to reread before touching code — this project's grading bar is explicitly "explain it without Google," not just "make the checker pass."
