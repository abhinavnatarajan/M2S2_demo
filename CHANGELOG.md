# Changelog

## 0.2.0

Migrate statistics generation to chalc 15 and use Polars throughout the Python
pipeline and plotting notebooks. The project version is now 0.2.0.

### Scope

The statistics-generation pipeline now uses chalc 15.0.0 and phimaker 0.5.0.
The mathematical choice of morphism, the 38 persistence statistics, the persisted
feature names, and the downstream analysis logic are preserved.

No research datasets or figures were generated, overwritten, or converted.
The untracked `leon_data/` directory was left untouched. No changes were committed.
The project virtual environment was synchronized to the updated dependency lock.

### Source changes

| File | Change |
| --- | --- |
| `pyproject.toml` | Require `chalc~=15.0`, set version 0.2.0, and remove pandas. Polars was already a direct dependency. |
| `utils/generate_stats.py` | Import morphisms from `chalc.morphisms`; replace `KChromaticQuotient` with `KChromaticGluing`; consume `(name, module)` pairs from six-pack iteration; extract barcodes with `module.intervals(degree)`. |
| `utils/stat_schema.py` | Define the pipeline's stable `DiagramName` type and the mapping from chalc's full module names to existing statistic prefixes. |
| `utils/read_stats.py` | Import the stable pipeline `DiagramName` and return Polars frames with `table_format="polars"`. The Parquet and PyArrow modes retain their behavior. |
| `colorectal_cancer_dataset_analysis/classification_by_patient_pca.py` | Make the same type-import substitution; feature parsing, filtering, PCA, classifiers, and scoring are otherwise unchanged. |
| `tests/test_chalc15.py` | Add five small synthetic tests covering numerical extraction, degree limits, persistence edge cases, categorical dtypes, Parquet compatibility, and CSV generation/resume. |
| `tests/test_polars_migration.py` | Add seven synthetic tests for readers, existing Arrow partitions and resume, ABM dataframe transformations, plotting inputs, legacy HDF5, and notebook helper functions. |
| `utils/parallel.py` | Use spawned worker processes and a module-level initializer, avoiding inherited Polars thread locks. |
| `utils/plotting.py` | Use Polars filtering and columns for the weighted violin plot. |
| `abm_dataset_analysis/generate_stats.py` | Use Polars for CSV loading, metadata joins, and macrophage labels. |
| `abm_dataset_analysis/eee_clustering.py` | Replace index/tuple-column dataframes with Polars frames and explicit metadata columns, preserving the analysis formulas and clustering settings. |
| Plotting notebooks | Convert dataframe operations to Polars; preserve all saved outputs and execution counts. |
| `utils/legacy_results.py` | Read the original fixed-format HDF5 result tables directly with PyTables into Polars, without rewriting the archive. |

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

Worker processes now use `spawn`, following the
[Polars multiprocessing guidance](https://docs.pola.rs/user-guide/misc/multiprocessing/).
Call multiprocessing entry points from a script protected by
`if __name__ == "__main__":`; worker initializers must be pickleable.

The old plotting notebooks retain their historical input paths and analysis
choices. Their code cells were converted without executing them or updating any
stored figures. The legacy HDF5 reader supports numeric fixed-format result
frames with ordinary or MultiIndex rows; other HDF5 layouts are outside its scope.

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
   and resume matching; ABM metadata joins and macrophage splitting; duplicate
   pivot aggregation and missing values; explicit exclusion of metadata from
   model inputs; and the existing PCA formula. Model handoff and result grouping
   were checked with fixed labels, without running a clustering job.
7. **Notebook and archive checks.** All notebook code cells parse, and all saved
   outputs and execution counts match the original notebooks. Small helper
   functions were exercised on synthetic frames without executing figure-writing
   cells. The legacy reader passed numeric-block and MultiIndex checks; the
   test archive's bytes were unchanged. Pandas is absent from the installed
   environment and both lockfiles.

All twelve reusable tests passed:

```bash
uv run python -m unittest discover -s tests -v
```

The CSV smoke test uses the existing multiprocessing manager and requires local
IPC sockets. In the restricted execution sandbox that operation was blocked;
the suite passed when run with permission to use local IPC.

### Limits and existing issues

The full research pipeline, classifiers, clustering jobs, and figure notebooks
were not executed. Agreement on tiny synthetic examples is not a proof of
numerical identity on every dataset: chalc 15 also includes upstream persistence
bug fixes. Existing resume behavior is unchanged, including its lack of
dependency-version tracking for stored rows. Use a separate output directory
when comparing newly generated v15 results with archived paper results.

A targeted lint check also found a pre-existing undefined `FeatureParts`
reference in `utils/read_stats.py::discard_feature`. That function was not
modified by this migration. The classification script's own `FeatureParts`
implementation imports and parses the generated names successfully.

The ABM dataframe conversion removes the old dependency on the absent
`normalize_colname` helper: feature names are ordinary strings and metadata
columns are kept explicitly. Broader unfinished ABM analysis work remains
deferred. The PCA formula, feature-selection conditions, and KMeans parameters
were retained rather than revised as part of the dataframe conversion.
