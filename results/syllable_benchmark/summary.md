# Syllable-timing benchmark (D-033)

Choice rule: highest mean target r over day195_m172, day195_m165 (free, forced TTS), over (arm, method); ties within 0.01 go to the simpler candidate (A/current < A/nuclei < A/sylber < B/current < B/nuclei < B/sylber).

**Winner: arm B, current** (means: A/current 0.1938, A/nuclei 0.1926, A/sylber 0.1711, B/current 0.3158, B/nuclei 0.2509, B/sylber 0.2652; tied within tolerance: B/current)

| cell | age (d) | arm | method | variant | TTS | onsets <= 100 ms | span onset err (ms) | target r | chance r | oracle r (n) | r on oracle trials |
|---|---|---|---|---|---|---|---|---|---|---|---|
| day195_m172 | 23 | A | current | free | forced | 0.369 | 80 | 0.194 | 0.061 | 0.460 (184) | 0.195 |
| day195_m172 | 23 | A | current | count_constrained | forced | 0.364 | 80 | 0.189 | 0.061 | 0.460 (184) | 0.190 |
| day195_m172 | 23 | A | nuclei | free | forced | 0.346 | 80 | 0.191 | 0.061 | 0.460 (184) | 0.194 |
| day195_m172 | 23 | A | nuclei | count_constrained | forced | 0.336 | 80 | 0.182 | 0.060 | 0.460 (184) | 0.186 |
| day195_m172 | 23 | A | sylber | free | forced | 0.285 | 100 | 0.171 | 0.058 | 0.460 (184) | 0.176 |
| day195_m172 | 23 | A | sylber | count_constrained | forced | 0.277 | 110 | 0.167 | 0.059 | 0.460 (184) | 0.173 |
| day195_m172 | 23 | B | current | free | forced | 0.589 | 80 | 0.315 | 0.054 | 0.460 (184) | 0.320 |
| day195_m172 | 23 | B | current | count_constrained | forced | 0.582 | 90 | 0.312 | 0.055 | 0.460 (184) | 0.318 |
| day195_m172 | 23 | B | nuclei | free | forced | 0.468 | 90 | 0.242 | 0.051 | 0.460 (184) | 0.245 |
| day195_m172 | 23 | B | nuclei | count_constrained | forced | 0.450 | 120 | 0.234 | 0.048 | 0.460 (184) | 0.234 |
| day195_m172 | 23 | B | sylber | free | forced | 0.383 | 130 | 0.266 | 0.052 | 0.460 (184) | 0.272 |
| day195_m172 | 23 | B | sylber | count_constrained | forced | 0.388 | 140 | 0.260 | 0.053 | 0.460 (184) | 0.267 |
| day195_m172 | 23 | A | current | free | envelope | 0.480 | 70 | 0.230 | 0.055 | 0.428 (124) | 0.240 |
| day195_m165 | 30 | A | current | free | forced | 0.364 | 90 | 0.194 | 0.061 | 0.460 (184) | 0.202 |
| day195_m165 | 30 | A | current | count_constrained | forced | 0.359 | 90 | 0.192 | 0.061 | 0.460 (184) | 0.200 |
| day195_m165 | 30 | A | nuclei | free | forced | 0.354 | 100 | 0.194 | 0.061 | 0.460 (184) | 0.204 |
| day195_m165 | 30 | A | nuclei | count_constrained | forced | 0.337 | 100 | 0.181 | 0.060 | 0.460 (184) | 0.187 |
| day195_m165 | 30 | A | sylber | free | forced | 0.255 | 120 | 0.171 | 0.060 | 0.460 (184) | 0.179 |
| day195_m165 | 30 | A | sylber | count_constrained | forced | 0.249 | 120 | 0.167 | 0.058 | 0.460 (184) | 0.174 |
| day195_m165 | 30 | B | current | free | forced | 0.582 | 110 | 0.317 | 0.053 | 0.460 (184) | 0.327 |
| day195_m165 | 30 | B | current | count_constrained | forced | 0.581 | 120 | 0.313 | 0.057 | 0.460 (184) | 0.322 |
| day195_m165 | 30 | B | nuclei | free | forced | 0.497 | 110 | 0.260 | 0.052 | 0.460 (184) | 0.268 |
| day195_m165 | 30 | B | nuclei | count_constrained | forced | 0.472 | 120 | 0.243 | 0.052 | 0.460 (184) | 0.250 |
| day195_m165 | 30 | B | sylber | free | forced | 0.333 | 160 | 0.264 | 0.053 | 0.460 (184) | 0.276 |
| day195_m165 | 30 | B | sylber | count_constrained | forced | 0.339 | 170 | 0.256 | 0.053 | 0.460 (184) | 0.264 |
| day195_m165 | 30 | A | current | free | envelope | 0.486 | 90 | 0.235 | 0.058 | 0.428 (124) | 0.245 |
| day195_m193 | 2 | A | current | free | forced | 0.430 | 50 | 0.235 | 0.058 | 0.460 (184) | 0.239 |
| day195_m193 | 2 | A | current | count_constrained | forced | 0.414 | 50 | 0.226 | 0.058 | 0.460 (184) | 0.229 |
| day195_m193 | 2 | A | nuclei | free | forced | 0.418 | 50 | 0.231 | 0.059 | 0.460 (184) | 0.235 |
| day195_m193 | 2 | A | nuclei | count_constrained | forced | 0.381 | 50 | 0.210 | 0.059 | 0.460 (184) | 0.216 |
| day195_m193 | 2 | A | sylber | free | forced | 0.292 | 80 | 0.199 | 0.060 | 0.460 (184) | 0.205 |
| day195_m193 | 2 | A | sylber | count_constrained | forced | 0.285 | 80 | 0.192 | 0.061 | 0.460 (184) | 0.197 |
| day195_m193 | 2 | B | current | free | forced | 0.685 | 60 | 0.359 | 0.053 | 0.460 (184) | 0.369 |
| day195_m193 | 2 | B | current | count_constrained | forced | 0.662 | 60 | 0.339 | 0.055 | 0.460 (184) | 0.349 |
| day195_m193 | 2 | B | nuclei | free | forced | 0.549 | 50 | 0.281 | 0.055 | 0.460 (184) | 0.294 |
| day195_m193 | 2 | B | nuclei | count_constrained | forced | 0.499 | 70 | 0.252 | 0.051 | 0.460 (184) | 0.266 |
| day195_m193 | 2 | B | sylber | free | forced | 0.328 | 130 | 0.299 | 0.052 | 0.460 (184) | 0.306 |
| day195_m193 | 2 | B | sylber | count_constrained | forced | 0.320 | 130 | 0.280 | 0.050 | 0.460 (184) | 0.292 |
| day195_m193 | 2 | A | current | free | envelope | 0.570 | 40 | 0.279 | 0.057 | 0.428 (124) | 0.281 |
| day193_m165 | 28 | A | current | free | forced | 0.310 | 120 | 0.230 | 0.059 | 0.462 (248) | 0.231 |
| day193_m165 | 28 | A | current | count_constrained | forced | 0.303 | 120 | 0.227 | 0.060 | 0.462 (248) | 0.229 |
| day193_m165 | 28 | A | nuclei | free | forced | 0.300 | 120 | 0.227 | 0.059 | 0.462 (248) | 0.228 |
| day193_m165 | 28 | A | nuclei | count_constrained | forced | 0.282 | 125 | 0.221 | 0.059 | 0.462 (248) | 0.222 |
| day193_m165 | 28 | A | sylber | free | forced | 0.221 | 150 | 0.209 | 0.059 | 0.462 (248) | 0.209 |
| day193_m165 | 28 | A | sylber | count_constrained | forced | 0.212 | 150 | 0.206 | 0.060 | 0.462 (248) | 0.207 |
| day193_m165 | 28 | B | current | free | forced | 0.490 | 130 | 0.321 | 0.051 | 0.462 (248) | 0.337 |
| day193_m165 | 28 | B | current | count_constrained | forced | 0.494 | 130 | 0.319 | 0.053 | 0.462 (248) | 0.332 |
| day193_m165 | 28 | B | nuclei | free | forced | 0.412 | 120 | 0.259 | 0.046 | 0.462 (248) | 0.269 |
| day193_m165 | 28 | B | nuclei | count_constrained | forced | 0.407 | 140 | 0.253 | 0.047 | 0.462 (248) | 0.261 |
| day193_m165 | 28 | B | sylber | free | forced | 0.309 | 190 | 0.262 | 0.047 | 0.462 (248) | 0.270 |
| day193_m165 | 28 | B | sylber | count_constrained | forced | 0.318 | 190 | 0.265 | 0.047 | 0.462 (248) | 0.273 |
| day193_m165 | 28 | A | current | free | envelope | 0.395 | 100 | 0.267 | 0.054 | 0.438 (195) | 0.272 |

TTS side (forced alignment + maximal onset): syllable count = CMUdict count in kunz_50words: 0.992 (250/250 aligned); t15_day193: 0.995 (432/432 aligned); t15_day195: 0.976 (294/294 aligned).
