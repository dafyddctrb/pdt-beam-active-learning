# Third-party code

Several modules in `updating/` are based on software of the
**Engineering Risk Analysis (ERA) Group, Technical University of Munich**
(<https://www.cee.ed.tum.de/era/software/>, <https://github.com/ERA-Software>).
The original author headers are kept in each file.

| File | Origin | Changes |
|---|---|---|
| `updating/ERADist.py` | ERA distribution class (`ERADist`), version 2021-03 | none |
| `updating/ERANataf.py` | ERA Nataf transformation (`ERANataf`), version 2023-05 | none |
| `updating/AL_aBUS_SuS.py` | ERA adaptive BUS with Subset Simulation (`aBUS_SuS`), version 2021-03 | GP surrogate prediction model, active learning at each subset level |
| `updating/AL_aCS_aBUS.py` | ERA adaptive conditional sampling for aBUS (`aCS_aBUS`), version 2018-05 | Likelihood evaluated through the surrogate prediction model. All chains advanced in parallel, one step at a time, with the proposal scale adapted after about every 100 candidates. Current chain states not re-evaluated. |

`ERADist` and `ERANataf` are released by the ERA Group under the MIT License, and `aBUS_SuS` and
`aCS_aBUS` are included under the same licence with the group's permission. Like the rest of
this repository, all four files are distributed under the MIT License (see `LICENSE`).

## References

- Straub, D. & Papaioannou, I. (2015). Bayesian updating with structural reliability methods. *Journal of Engineering Mechanics*, 141(3).
- Betz, W., Papaioannou, I., Beck, J. L. & Straub, D. (2018). Bayesian inference with Subset Simulation: strategies and improvements. *Computer Methods in Applied Mechanics and Engineering*, 331, 72–93.
- Papaioannou, I., Betz, W., Zwirglmaier, K. & Straub, D. (2015). MCMC algorithms for Subset Simulation. *Probabilistic Engineering Mechanics*, 41, 83–103.
