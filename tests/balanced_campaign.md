# Bounded balanced-parameter campaign

`balanced_campaign.py` is an exploratory public-API screen. It leaves production defaults untouched. The frozen six-scene archive pilot is scene-disjoint within this campaign but reuses images from earlier archives; it is not a fresh external holdout. Its 128-pixel timing is not a full-frame speed claim.

Prepare once, before looking at denoised outputs:

```sh
python3 tests/balanced_campaign.py prepare \
  --source /path/to/fixtures128 \
           /path/to/fixtures512 \
  --out artifacts/balanced/fixtures128
```

This writes 60 cases: six scenes, sigma 5/10/25/50/75 and two deterministic new AWGN seeds. The split is search `house`/`DIV2K-0805`, validation `barbara`/`DIV2K-0822`, and sealed `peppers256`/`DIV2K-0840`. Crops never cross scenes or resize source content. Gray and genuine RGB are distinct; MCWNNM Gray cells are explicitly unsupported.

On a pinned single-CPU host, start with sigma 25 and one seed:

```sh
taskset -c 0 python3 tests/balanced_campaign.py run \
  --plugin /path/to/libnss.so --fixtures /path/to/fixtures128 \
  --out /path/to/new-pilot --split search --sigmas 25 --noise-repeat 0 \
  --repeats 1 --budget-seconds 1800 --twsc-timeout 120
```

The runner evaluates 30 recipes, including current public omissions and explicit parameter alternatives. NLM candidates calibrate strength as a fraction of known sigma. BM3D `basic-final` measures both stages. NLH legacy recipes replay explicit legacy parameters on the current model. NCSR invokes the actual public NCSR implementation; it does not silently substitute the separate CPU HQ path.

The worker holds six source frames, loops them into unique frame numbers, disables output and intermediate BM3D caches, warms up, and times only frame requests. Source fills must remain zero and warm/final static output arrays must match exactly. Output files, hashes, unclipped float PSNR/SSIM and per-plane quality, residual response, public frame properties, backend and immutable host identity accompany every successful row. Timings qualify only for the verified CPU0/CPU1 SMT topology with at least ten sibling counter ticks, at most 1% sibling activity and no steal. Very short measurements may therefore have quality evidence but unqualified timing; increase `--target-seconds`/`--max-frames` in a new output directory when needed.

`--budget-seconds` caps each invocation including worker startup/warmup. Each subprocess also has `--timeout` (180 seconds by default) and TWSC's separate timeout. No workload is silently resized. Repeating exactly the same command and output directory resumes only pending jobs. Terminal failures/timeouts are kept and are not retried silently. Different source, plugin, candidate, fixture, case, repeat or timing identities require a new directory. The raw JSONL is append-only; `completion.json` explicitly lists pending cells.

For expanded screening, omit `--sigmas` and `--noise-repeat`; use at least three repeats for serious timing comparisons. Results are paired by repeat with alternating candidate order. `summary.json` retains per-case quality deltas and qualified paired speed ratios; `report.md` is a compact view. The default decision remains “no qualified replacement” until the full plan's validation is complete.

Select candidates using search results, then run the validation split with a frozen selection file:

```json
{
  "candidates_sha256": "SHA256_OF_FROZEN_candidates.json",
  "algorithms": {
    "NLM": ["h06-spatial"],
    "BM3D": ["g16-step8"]
  }
}
```

Pass it as `--finalists /path/to/selection.json`. Entries are illustrative, not recommendations. Algorithms without a selected candidate run only their current default. The sealed split refuses to run without a selection file whose recipe hash matches the frozen configuration. Keep the selection and its search/validation rationale in the campaign evidence before opening sealed results; do not change a finalist after viewing sealed outputs.

The original plan still requires larger scene-diverse images, native YUV, moving clips with independent temporal noise, unequal-channel noise, explicit versus blind estimation, texture inspection and memory constraints. This screen does not assert those gates passed. A candidate must satisfy the proposed 1.2× speed, median ≤0.2 dB/worst ≤0.5 dB PSNR loss and worst ≤0.005 SSIM loss criteria on appropriate validation/sealed data, with no material low-noise or texture regressions. Under-denoising defaults can require a quality-first recipe with an explicit speed cost.

Use the standalone assessor for format/noise decisions after each frozen run:

```sh
python3 tests/balanced_assess.py \
  --results /path/to/search-run /path/to/refinement-run /path/to/validation-run \
  --fixtures /path/to/fixtures128 /path/to/fixtures256 \
  --cold-baselines /path/to/twsc-default-quality \
  --out /path/to/new-assessment
```

It computes per-format, per-noise and all-noise median/worst PSNR and SSIM deltas, worst per-plane PSNR delta, low-noise worse-than-noisy cases, eligible pair counts, latency and plugin-reported resource peaks. Optional `--max-frame-ms` and `--max-memory-mb` express practical limits. It keeps plugins and dataset splits separate, does not count duplicate case observations as independent scenes, and rejects duplicate raw result inputs. A cold TWSC `.f32.json` reference is accepted only when explicitly marked `warmup_performed=false`; it supplies a quality bound after plugin/input hashes match and never supplies a timing ratio.

An assessment label distinguishes measured speed/quality thresholds from a quality-improving cost tradeoff or a missing-baseline result. It always includes all eight algorithm decisions, and does not promote public defaults or freeze sealed finalists automatically. Qualifying measured numbers do not replace texture review, motion checks or a sufficiently representative holdout.
