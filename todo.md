# Colorectal cancer
- One stratified $K$-fold cross-validation diagnostic
    - Split the data into $K$ subsets using a fixed shuffled split.
    - Train $K$ models. Model $i$ is trained using all subsets except $i$, and tested on subset $i$.
    - Record the training and test score from every fold.
    - Record one score from the concatenated out-of-fold predictions.
- For grouped permutation importance:
    - Fit one final model on the complete patient dataset.
    - Jointly permute the rows of all features within each cell group and score the fitted model without refitting it.
    - Record the distribution of score decreases across permutations for every cell group.

Recommended: 30 permutations with Brier loss.

# ABM
- Zero imputation and standardization?
- clustering stability assessment by ten seeded runs and pairwise ARI
- ARI against qualitative labels
