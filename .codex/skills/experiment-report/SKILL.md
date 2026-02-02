# experiment-report

Purpose: create or update a reproducible experiment log in `docs/experiments/`.

When to use:
- After implementing a hypothesis and running evaluation.

Steps:
1) Copy `docs/experiments/_template.md` to `docs/experiments/<id>.md` if it does not exist.
2) Fill in hypothesis, change summary, risk, and commands executed.
3) Record required metrics: ROI, total stake, n_bets, test period, max drawdown.
4) If rolling, add design vs eval windows and paired delta vs baseline.
5) Record the ROI definition explicitly: ROI = profit / stake, profit = return - stake.
6) Note pooled vs step14 sign mismatch and preferred ROI for decisions.
7) List artifacts paths (metrics.json, comparison.json, reports).

Output:
- Completed `docs/experiments/<id>.md` compliant with the template.
