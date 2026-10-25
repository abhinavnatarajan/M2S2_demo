# Changelog

## 0.2.0

Migrate statistics generation to chalc 15 and use Polars throughout the Python
pipeline and plotting notebooks. The project version is now 0.2.0.

### Scope

The statistics-generation pipeline now uses chalc 15.0.0 and phimaker 0.5.0.
The mathematical choice of morphism, the 38 persistence statistics, the persisted
feature names are preserved. The ABM analysis changes are described below.

No research datasets or figures were generated, overwritten, or converted.
The untracked `leon_data/` directory was left untouched. No changes were committed.
The project virtual environment was synchronized to the updated dependency lock.

### Source changes

| File | Change |
| --- | --- |
| `pyproject.toml` | Require `chalc~=15.0`, set version 0.2.0, and remove pandas. Polars was already a direct dependency. |
| `utils/generate_stats.py` | Import morphisms from `chalc.morphisms`; replace `KChromaticQuotient` with `KChromaticGluing`; consume `(name, module)` pairs from six-pack iteration; extract barcodes with `module.intervals(degree)`. |
| `utils/stat_schema.py` | Define stable diagram names, the chalc module-name mapping, and the shared `FeatureParts` parser. |
| `utils/read_stats.py` | Read Parquet/Arrow/Polars tables and provide the shared feature selection and pivot used by both analyses. |
| `colorectal_cancer_dataset_analysis/classification_by_patient_pca.py` | Delegate feature parsing and reading to shared helpers, including the corrected IQR selector. PCA, classifiers, and scoring are unchanged. |
| `tests/test_chalc15.py` | Add five small synthetic tests covering numerical extraction, degree limits, persistence edge cases, categorical dtypes, Parquet compatibility, and CSV generation/resume. |
| `tests/test_polars_migration.py` | Check Arrow partitions/resume, ABM metadata transformations, plotting inputs, and the retained ABM notebook helper. |
| `tests/test_eee_clustering.py` | Add thirteen synthetic checks for shared filtering, no-PCA KMeans, deterministic grouped importance for three partition scores, label isolation, and output handling. |
| `utils/parallel.py` | Use spawned worker processes and a module-level initializer, avoiding inherited Polars thread locks. |
| `utils/plotting.py` | Use Polars filtering and columns for the weighted violin plot. |
| `abm_dataset_analysis/generate_stats.py` | Use Polars for CSV loading, metadata joins, and macrophage labels. |
| `abm_dataset_analysis/eee_clustering.py` | Remove PCA; use zero imputation, optional standardization, and seeded KMeans. Add ARI, NMI, and Binder-loss evaluation against EEE annotations and importance per cell group/diagram/dimension. |
| Plotting notebooks | Convert dataframe operations to Polars; preserve all saved outputs and execution counts. |

Singles and pairs still use `KChromaticInclusion(filt, 1)`. Triples and larger
combinations use `KChromaticGluing(filt, number_of_labels - 1)`, the renamed
counterpart of the previous quotient construction. The existing helper name
`get_stats_k_chromatic_quotient` is retained for callers.

The requested maximum diagram dimension is now passed to
`sixpack(max_homological_degree=...)`. Previously the code computed an unrestricted
six-pack and selected degrees afterward. Tests compare capped computations with
unrestricted v15 computations, including kernel and relative modules whose
internal constructions involve degree shifts.

Each degree is queried with an integer. In v15, passing a sequence to `intervals`
returns a dictionary keyed by degree, whereas the old code expected a list of
arrays. Explicit integer queries preserve the list expected by the statistics
helper and include empty degrees.

The mapping retained in the output schema is:

| chalc 15 module | Persisted prefix |
| --- | --- |
| `kernel` | `ker` |
| `domain` | `dom` |
| `image` | `im` |
| `codomain` | `cod` |
| `cokernel` | `cok` |
| `relative` | `rel` |

Thus names such as `ker0-avg_birth` remain unchanged. With the default degrees
0 and 1, each combination still has 6 × 2 × 38 = 456 statistic columns, plus
`codomain` and dataset metadata. Existing Parquet statistics can still be read;
this migration does not require rewriting them.

`utils/persistent_stats.py` is unchanged. In particular, the existing handling
of zero-length, empty, and infinite bars remains in place. Although v15 removes
zero-persistence intervals earlier, the pipeline already removed those intervals
before calculating statistics.

### Polars migration and API changes

CSV loading, label filtering/counting, point extraction, resume checks, and
statistics-table creation now use Polars. Parquet output still uses PyArrow's
partitioned writer with the same partition keys, compression, and resume policy.
Tables cross the Polars/Arrow boundary directly, without a pandas conversion.
Custom statistics data loaders must now return `polars.DataFrame` instances.

Point labels are converted to Polars categorical values before filtering and
grouping, and retain that representation when dispatched to workers. Colour
extraction reuses those codes without converting labels back to strings.
String partition metadata retains categorical encoding before Arrow conversion;
patient IDs and other integer metadata keep their integer types because Polars
categorical values are strings. Additional metadata columns retain their inferred
types, matching the scope of the original partition-column casts.

`read_stats(..., table_format="polars")` replaces the former `"pandas"` mode.
The default `"parquet"` mode and `"pyarrow"` mode are preserved. ABM sample/index
metadata are ordinary columns and are explicitly excluded from numeric model
inputs. Feature names use `<statistic>-<cell/group>` strings in memory instead
of tuple labels; persisted statistics retain their established schema.

The colorectal cross-validation splitter and `DummyClassifier.fit` now receive
NumPy arrays explicitly, matching scikit-learn's type annotations. The grouped
PCA pipeline still receives Polars frames to retain feature names and grouping.
`get_n_splits()` is called without its unused data arguments. Format-specific
`read_stats` overloads expose the correct Parquet, Arrow, or Polars return type
to type checkers without changing runtime behavior.

Worker processes now use `spawn`, following the
[Polars multiprocessing guidance](https://docs.pola.rs/user-guide/misc/multiprocessing/).
Call multiprocessing entry points from a script protected by
`if __name__ == "__main__":`; worker initializers must be pickleable.

The retained ABM plotting notebooks keep their historical input paths and analysis
choices. Their code cells were converted without executing them or updating any
stored figures. Historical PCA comparisons in those notebooks have not been
regenerated for the refurbished clustering pipeline.

### ABM clustering refurbishment

Both analyses now call `read_feature_data` for the same lazy Parquet scan,
cell-group filtering, diagram/statistic filtering, canonical cell-group naming,
and pivot. `FeatureParts` lives in `utils/stat_schema.py`. The colorectal module
continues to expose its imported `FeatureParts` name for existing callers.
Metadata remain separate from model features. Duplicate sample/cell-group rows
are rejected, matching colorectal pivot behavior, rather than averaged silently.

The common filters discard codomain/relative diagrams, cokernel degree zero,
redundant domain/image degree-zero birth/midpoint/length summaries, IQR, non-domain
single-cell diagrams, and domain diagrams for multiple cell types. The previous
colorectal IQR regex ended in `-iqr$`, which did not select names such as
`ker1-iqr_birth`. The shared selector now uses the prefix `...-iqr` without an
end anchor, covering current and future IQR suffixes. This changes the colorectal
feature set to apply its documented IQR rule.

ABM PCA code and its three CLI flags have been removed. The model is now
constant-zero imputation, per-feature standardization, and KMeans. Scaling can
be disabled with `--no-standardize`; there is no dimension reduction. KMeans
uses 20 initializations and an explicit `--clustering-seed` (default 0).

EEE annotations are excluded from fitting and used only for ARI, NMI, and Binder
loss evaluation on individual simulation samples. NMI explicitly uses the
arithmetic entropy normalizer. Binder loss uses equal split/merge weights and is
normalized by the number of unordered sample pairs: the fraction of pairs whose
co-clustering status differs, equivalently `1 - Rand index`. Zero Binder loss is
best; it is not adjusted for chance.

For each `(cell group, diagram, dimension)`, all retained statistic columns are
shuffled together using one row permutation. The imputer, scaler, and KMeans
remain fitted and fixed. ARI and NMI importance use `baseline - permuted`;
Binder importance uses `permuted_loss - baseline_loss`. Thus positive importance
means permutation worsens agreement for all metrics; negative values are retained.
All groups and metrics share the same pre-generated permutations, controlled by
`--permutations-seed`. Each permutation makes one prediction call, and all three
scores use those predictions. `--num-permutations` defaults to 100.

The JSON result retains `labels` and parameter-pair `results`, and adds sample
IDs, `final_model_scores` keyed by `ari`, `nmi`, and `binder_loss`, preprocessing
and metric conventions, and a `feature_importances` list. Each entry identifies
the cell group, diagram, dimension, and column names. Its `importances`, `mean`,
and `std` mappings contain entries for all three metrics. The CLI refuses to
overwrite an existing result file. The former interim singular
`final_model_score`/`score_name` fields are replaced by `final_model_scores`.

Example command (not executed on research data during this work):

```bash
uv run python -m abm_dataset_analysis.eee_clustering \
    --stats-dir abm_dataset_analysis/stats \
    --output-file abm_dataset_analysis/results/eee_ari_importance_v2.json \
    --k 3 --num-permutations 100
```

These are final-model reliance measures on the observed data, as in the
colorectal final-model importance calculation. They are not held-out predictive
performance or causal effects. Correlated groups can mask each other, and groups
with different numbers of statistics are different-sized perturbations.

Alternative scores all require a reference partition unless they are internal
geometry criteria:

| Score | Interpretation and tradeoff |
| --- | --- |
| ARI (implemented) | Pairwise agreement adjusted for chance; invariant to cluster-label numbering. |
| NMI (implemented) | Normalized shared information; not adjusted for chance. AMI is the chance-adjusted alternative. |
| Variation of information | Conditional-entropy loss measuring splits and merges; zero is best. Loss importance uses `permuted_loss - baseline_loss`. |
| Binder loss (implemented) | Equal split/merge weights, normalized by the number of pairs: `1 - Rand index`. Asymmetric penalties are not implemented. |
| Negative inertia | No reference labels needed; evaluates distances to the fixed fitted centers, with dependence on units/scaling. |

See [scikit-learn clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation),
[permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html),
and [Wade and Ghahramani on Binder and VI losses](https://arxiv.org/abs/1505.03339).

### Dependency locks

`uv.lock` now resolves chalc 15.0.0 (previously 14.0.0), phimaker 0.5.0
(previously 0.3.0), and project version 0.2.0. Pandas and its pytz dependency have
been removed from both lockfiles and the project environment.

`pylock.toml` was regenerated with its documented command:

```bash
uv export --format pylock.toml --all-groups --no-emit-project -o pylock.toml
```

Its larger diff includes pre-existing differences from `uv.lock`: notebook and
development dependencies such as marimo, plotly, and pyrefly; pyarrow 25.0.1
instead of 20.0.0; and jedi 0.19.2 instead of 0.20.0. These versions were already
in `uv.lock`. The export also narrows the Python marker to `==3.13.*`, consistent
with the project's existing `requires-python = "~=3.13.0"`, and filters wheel
entries accordingly.

### Validation

Only generated synthetic inputs were used, with at most 18 points per cloud.
CSV inputs, generated Parquet files, logs, and the temporary v14 baseline were
kept outside research-data and figure directories.

1. **v14-to-v15 numerical comparison.** Before changing the environment, captured
   v14.0.0 statistics for a deterministic 18-point cloud with three labels
   (`numpy.random.default_rng(20261020)`, coordinates uniform on `[0, 10)`).
   Compared singles, pairs, and triples for `alpha`, `delaunay_cech`, and
   `delaunay_rips`, in degrees 0–2. All nine cases had identical sets of
   684 statistic names, and all 6,156 values matched within `rtol=1e-10`,
   `atol=1e-12`. Repeating the comparison after the Polars conversion and
   uninstalling pandas gave the same result. This was a one-time comparison
   with the installed old version; the reusable tests require only v15.
2. **Degree-limit checks.** The reusable suite compares extracted statistics to
   unrestricted v15 six-packs for all three algorithms, all three combination
   sizes, and maximum degrees 0, 1, and 2: 27 configurations.
3. **Edge cases and analysis compatibility.** Checked empty diagrams, finite
   plus infinite bars, discarded diagonal bars, unused categorical labels,
   Parquet round trips, and analysis parsing of the existing feature prefixes.
4. **CSV-to-Parquet smoke test.** One synthetic CSV, one worker, and all seven
   nonempty subsets of three labels produced seven rows and 460 columns
   (456 statistics, `codomain`, and three sample metadata columns). Categorical
   string metadata and integer IDs were checked at the Arrow writer boundary
   and after reading; categorical labels were also checked before worker dispatch.
   A second run using resume
   preserved all generated Parquet bytes.
5. **Environment and static checks.** `uv pip check` reported compatible installed
   packages; targeted fatal-error lint checks passed for the new tests, schema,
   generator, and classification script; `git diff --check` and
   `uv lock --check --offline` passed. Both dataset statistics-generator modules
   and the colorectal classification module imported successfully.
6. **Polars compatibility.** Checked dictionary-encoded legacy Arrow partitions
   and resume matching; ABM metadata joins and macrophage splitting; explicit
   exclusion of metadata from model inputs; and the revised shared pivot/filtering.
7. **Notebook and archive checks.** All notebook code cells parse, and all saved
   outputs and execution counts match the original notebooks. Small helper
   functions were exercised on synthetic frames without executing figure-writing
   cells. Pandas is absent from the installed environment and both lockfiles.

The ABM-specific tests exercise only eight-sample synthetic datasets. They check
identical ABM/colorectal feature tables, joint permutations including missingness,
correct grouping, reproducibility, annotation isolation, and retained negative
importances for all three metrics. A hand-counted six-pair example verifies Binder
normalization. Prediction-call counts verify that adding metrics does not add
prediction passes. A synthetic future IQR statistic verifies prefix filtering.
The project passes Pyrefly and the affected modules pass targeted fatal-error lint checks.
The user corrected the numeric type guard in `weighted_violinplot`; the plotting
regression test is included in the full suite.

Final validation: all 22 regression tests passed, including 13 ABM checks and the
plotting test. Full-project Pyrefly checking includes the notebooks and tests and
reports zero errors with versions 1.2.0 and 1.4.0, with two scoped suppressions
(the existing plotting scalar overload and the scikit-learn import described
below). Version 1.2.0 retains two existing warnings about redundant `float()`
calls in the colorectal visualization scripts; version 1.4.0 instead reports
two warnings about missing SciPy type stubs in the PCA and plotting helpers.
The earlier four-module check missed diagnostics elsewhere: predictions are now
explicitly converted to NumPy before JSON serialization, test unions are narrowed,
notebook label alignment rejects unsupported cluster counts, and tick labels are
explicitly strings. The existing colorectal PCA helper now qualifies its nested
`GroupModel` annotation. Its `validate_data` import has a narrowly scoped,
documented suppression because Pyrefly's bundled scikit-learn stubs omit this
public API; the installed scikit-learn implementation was checked directly.
A separate 12-sample colorectal check with two cross-validation folds and two
permutations compared the previous dataframe calls against the explicit NumPy
calls. Every result field was exactly unchanged, including cross-validation
scores, baseline scores, final scores, and grouped importance arrays.
Only tiny synthetic datasets were used.

Run the tests with:

```bash
uv run python -m unittest discover -s tests -v
```

The CSV smoke test uses the existing multiprocessing manager and requires local
IPC sockets. In the restricted execution sandbox that operation was blocked;
the suite passed when run with permission to use local IPC.

### Limits and existing issues

The full research pipeline and figure notebooks were not executed. Classifier
and clustering runs used only tiny synthetic inputs. Agreement on these is not a proof of
numerical identity on every dataset: chalc 15 also includes upstream persistence
bug fixes. Existing resume behavior is unchanged, including its lack of
dependency-version tracking for stored rows. Use a separate output directory
when comparing newly generated v15 results with archived paper results.

The previously undefined `FeatureParts` reference in `utils/read_stats.py` is
resolved by using the shared parser. The retired ABM `normalize_colname` import
is no longer needed. The PCA-preservation and removed legacy-notebook tests
have been replaced with checks of the current pipeline and retained files.
