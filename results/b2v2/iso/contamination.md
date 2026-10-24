# D-080 post-hoc: acoustic-contamination checks

(a) Cross-condition generalization (chance 14.3%):

| train -> test | accuracy | null | p |
|---|---|---|---|
| listening->imaginedListening | 33.8% | 14.5% | 0.002 |
| imaginedListening->listening | 45.7% | 14.0% | 0.002 |
| listening->reading | 17.9% | 14.2% | 0.13 |
| listening->imagined1stAuditory | 24.3% | 14.4% | 0.002 |
| listening->attempted | 45.0% | 14.2% | 0.002 |

(b) Latency (unsmoothed, 50 ms windows from the delay start):

- listening: first window above the null's 95th percentile at 60 ms; peak 65.7% at 500 ms
- reading: first window above the null's 95th percentile at 20 ms; peak 31.4% at 660 ms
