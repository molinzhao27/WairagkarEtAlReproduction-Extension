# Neural matching: position control (post hoc)

Paired attempted/imagined blocks present the same sentences in the same order, so the own attempted trial is
also the same-position trial. Non-paired blocks share positions but not sentences: if slow drift within blocks
drove the match, same-position trials would match there too. For every (imagined block, attempted block)
combination this ranks the same-position attempted trial among all 50 (DTW trajectory similarity, the primary
measure; PCA fitted on the two pairs not involved) and reports mean similarity by |position difference|.

| Imagined | Attempted | Kind | Same-position mean rank (chance 25.5) | Top-1 (chance 0.02) | sim, abs dpos 0 | sim, abs dpos 1 | sim, abs dpos 2 | sim, abs dpos 3 | sim, abs dpos 4-9 | sim, abs dpos >=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| I04 | A03 | paired | 6.76 | 0.40 | 0.245 | 0.184 | 0.183 | 0.182 | 0.185 | 0.177 |
| I04 | A06 | control | 24.88 | 0.02 | 0.194 | 0.195 | 0.203 | 0.194 | 0.197 | 0.192 |
| I04 | A11 | control | 26.26 | 0.04 | 0.164 | 0.164 | 0.169 | 0.167 | 0.163 | 0.165 |
| I04 | A13 | control | 26.88 | 0.00 | 0.174 | 0.186 | 0.176 | 0.180 | 0.178 | 0.176 |
| I07 | A03 | control | 27.76 | 0.00 | 0.162 | 0.169 | 0.167 | 0.169 | 0.170 | 0.169 |
| I07 | A06 | paired | 12.72 | 0.26 | 0.205 | 0.159 | 0.161 | 0.158 | 0.162 | 0.164 |
| I07 | A11 | control | 29.36 | 0.04 | 0.136 | 0.146 | 0.141 | 0.142 | 0.144 | 0.143 |
| I07 | A13 | control | 27.92 | 0.00 | 0.147 | 0.147 | 0.154 | 0.150 | 0.155 | 0.151 |
| I09 | A03 | control | 24.54 | 0.02 | 0.146 | 0.144 | 0.142 | 0.143 | 0.141 | 0.144 |
| I09 | A06 | control | 24.70 | 0.04 | 0.139 | 0.136 | 0.141 | 0.137 | 0.138 | 0.139 |
| I09 | A11 | paired | 9.52 | 0.42 | 0.198 | 0.153 | 0.149 | 0.150 | 0.151 | 0.152 |
| I09 | A13 | control | 24.34 | 0.00 | 0.151 | 0.147 | 0.147 | 0.153 | 0.143 | 0.149 |
| I12 | A03 | control | 25.64 | 0.00 | 0.161 | 0.162 | 0.162 | 0.164 | 0.163 | 0.161 |
| I12 | A06 | control | 25.48 | 0.02 | 0.146 | 0.145 | 0.143 | 0.146 | 0.144 | 0.147 |
| I12 | A11 | control | 27.56 | 0.02 | 0.166 | 0.170 | 0.171 | 0.176 | 0.171 | 0.170 |
| I12 | A13 | paired | 3.78 | 0.54 | 0.237 | 0.168 | 0.168 | 0.164 | 0.165 | 0.166 |

Control pooled (non-paired combos): mean rank 26.28, top-1 0.017 over 12 combos.
