# Colorectal cancer
- Repeated $K$-fold cross-validation
    - Split the data into $K$ subsets.
    - Train $K$ models. Model $i$ is trained using all subsets except $i$, and tested on subset $i$.
    - Using each observation's out-of-fold prediction, get a test accuracy (balanced accuracy, Brier loss, precision-recall AUC or ROC-AUC)
    - Repeat these steps $r$ times, report the average and standard deviation of the $r$ test accuracy scores.
- For grouped permutation importance:
    - For each repetition $i$ and each fold $j$, permute the rows within each cell group in the test fold only, and compute the new set of test accuracy scores. Compute drop in the chosen accuracy score in the repetition $i$, associated with the permutation $k$.
    - Perform the permutation step several times for each reptition $i$.
    - Report average accuracy loss across all $i$ and $k$.

Recommended : 50 repetitions, 30 permutations with Brier loss.
Report distribution of importance scores across repetitions, averaging the importance score within each repetition.
Report distribution of accuracies across repetitions.

# ABM
- Zero imputation and standardization?
- clustering stability assessment by ten seeded runs and pairwise ARI
- ARI against qualitative labels
