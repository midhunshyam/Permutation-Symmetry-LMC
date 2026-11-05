# Results

Raw results, one JSON object per line: one trained pair (or, for `clique`, one K-tuple) per line. The `summary*.md` files are generated tables; regenerate them with the analysis scripts rather than editing them. Non-finite values are stored as `null`.

## `mnist/`

Written by `scripts/run_mnist.py`, summarised by `scripts/analyze_mnist.py`. `main` is the protocol of every reported number (two hidden layers of width 256, 6 epochs).

| file | one line per | identifying fields |
|---|---|---|
| `methods_main.jsonl` | method × seed (8 × 8) | `method`, `seed` |
| `lambda_main.jsonl` | midpoint weight λ × seed (9 × 8), all `joint_midpoint` | `lam`, `seed` |
| `clique_main.jsonl` | clique size K × seed (4 × 4) | `K`, `seed` |

Fields of a pair record (`methods`, `lambda`):

| field | meaning |
|---|---|
| `method`, `seed` | method id (see `METHODS` in `lmc/mnist/experiment.py`) and seed-pair index |
| `lam` | midpoint weight λ for `joint_midpoint` and `frozen_anchor`; `null` otherwise |
| `config` | `hidden`, `n_hidden_layers`, `epochs`, `batch_size`, `lr`, `weight_decay`, `shuffle_seed` |
| `train_seconds` | wall-clock training time for the pair (CPU, 4 threads); depends on machine load, and the `tna_pfn` and clique runs shared the CPU with other jobs |
| `a`, `b`, `midpoint` | test `loss`, `acc`, `ece` (15 bins) and `brier` of network A, network B and the network with parameters ½(θ_A + θ_B) |
| `curve` | `alphas`, `losses`, `accs`: test loss and accuracy at 25 evenly spaced α ∈ [0, 1] along (1 − α)θ_A + αθ_B; `midpoint_barrier`, `max_barrier` |
| `row_cosine` | mean cosine between corresponding hidden-layer weight rows of A and B, over trainable entries, without permutation |
| `param_cosine`, `param_distance` | cosine and Euclidean distance between A's and B's trainable parameter vectors |
| `logit_cka` | linear CKA between A's and B's test-set logits |
| `aligned` | after permuting B's hidden units to match A's: `curve`, `row_cosine`, `n_units_moved`, `n_hidden_units`; `null` for `w_asym` and `sigma_asym` |

"Trainable entries" are the weight entries that receive gradient: W-Asym's frozen positions and TNA-PFN's gradient-masked entries are left out. `aligned` matches the full effective weights.

A clique record holds `K`, `seed`, `lam`, `config`, `train_seconds`, `endpoint_accs` (test accuracy of each network), `average_acc` (test accuracy of the network whose parameters are the mean of all K) and `pair_barriers` (midpoint barrier of every pair).

## `fills/`

Written by `scripts/run_fills.py`, summarised by `scripts/analyze_fills.py`.

| file | experiment | identifying fields |
|---|---|---|
| `enumeration.jsonl` | 4-4-3 network: all 256 masks with one frozen entry per row, zero and Gaussian fill; plus the unmasked network (`fill = "unmasked"`, no `pattern`) | `pattern`, `fill`, `seed` |
| `sweep.jsonl` | 4-4-3 network: 26 fills × 3 masks | `pattern`, `fill`, `seed` |
| `hybrid.jsonl` | 4-4-3 network: `n_gaussian` of the 4 frozen entries Gaussian, the rest zero; `subset_index` numbers the subsets of that size | `pattern`, `n_gaussian`, `subset_index`, `seed` |
| `width.jsonl` | in_dim 64, 16 frozen entries per row, hidden 4–64, fixed design | `hidden`, `fill`, `seed` (with `design`, `in_dim`, `n_frozen`) |
| `grid.jsonl` | hidden 32: in_dim × n_frozen, 11 fills, both designs | `design`, `in_dim`, `n_frozen`, `fill`, `seed` (with `hidden`) |
| `grid_wide.jsonl` | hidden 64 and 128: in_dim ∈ {64, 128, 256} × n_frozen ∈ {8, 16, 32} with hidden ≤ in_dim, 7 fills, both designs | `design`, `hidden`, `in_dim`, `n_frozen`, `fill`, `seed` |
| `controls.jsonl` | hidden 32, fixed design: unmasked networks (`fill = "unmasked"`, `n_frozen = 0`), and a Gaussian fill drawn separately for B (`fill = "gaussian_unshared"`) | as `grid` |

`pattern` lists, for each row of W1, the input that row freezes. `design` is `fixed` (one mask and fill draw per cell) or `resampled` (a draw per seed pair); see `design_seeds` in `lmc/fills/experiment.py`.

Measurements in every fill record:

| field | meaning |
|---|---|
| `midpoint_barrier` | L(½(θ_A + θ_B)) − ½(L(θ_A) + L(θ_B)), test cross-entropy |
| `max_barrier`, `losses` | largest excess of the test loss over the chord, and the test loss at 11 evenly spaced α (absent in `enumeration`, which evaluates α ∈ {0, ½, 1} only) |
| `loss_a`, `loss_b`, `acc_a`, `acc_b`, `acc_mid` | test loss and accuracy of A and B, and test accuracy of the midpoint |
| `frozen_energy_share` | mean over rows of (squared norm of the row's frozen entries) / (squared norm of the row), averaged over A and B |
| `w1_cos`, `w1_cos_trainable` | mean cosine between corresponding rows of A's and B's W1, without permutation, over all entries (frozen ones included) / over trainable entries only |
