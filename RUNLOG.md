## Run 0 — baseline (blend.py)
- Method: rank 54 stock voices, naive 50/50 blend of top 2
- Top 5: af_nova 0.6230, zm_yunxia 0.6127, if_sara 0.6053, hf_beta 0.5875, jf_nezumi 0.5822
- Blend (af_nova + zm_yunxia): 0.6252
- BASELINE TO BEAT: 0.6252
## Run 1 — row-aware split-dimension search (search.py)
- Method: perturb only the 20 rows exercised by 5 fitness sentences (+/-2 neighborhood), separate step sizes for acoustic (:128) vs prosodic (128:) halves, fitness = mean similarity + 0.15*cross-sentence consistency - 0.1*naturalness penalty
- 150 iterations, 11 accepted, adaptive step (1/5 rule) shrunk from 0.04 to 0.0151
- Fitness: 0.7427 -> 0.7638
- Raw similarity (comparable to baseline): [paste your number from step 2]
- What I heard: [your honest listening notes]
