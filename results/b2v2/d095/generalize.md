# D-095: T15's voice decoder for other participants (text-only adaptation)

| participant / stage / arm | trials | top-1 | mean rank | null mean rank | p vs null | PER | words read right |
|---|---|---|---|---|---|---|---|
| t12 A transfer (attempted) | 200 | 11.5% | 15.1 | 20.4 | 1.1e-24 | 87.6% | 4.7% |
| t12 A random (attempted) | 200 | 2.5% | 20.0 | 20.5 | 3.5e-05 | 98.5% | 0.0% |
| t12 inner, zero-shot | 200 | 3.0% | 19.8 | 20.3 | 0.0061 | 97.1% | 0.3% |
| t12 B inner, adapted | 200 | 24.0% | 10.8 | 19.7 | 3.4e-27 | 80.9% | 8.8% |
| t16 A transfer (attempted) | 120 | 55.0% | 5.4 | 18.7 | 8.8e-21 | 64.3% | 27.4% |
| t16 A random (attempted) | 120 | 2.5% | 20.4 | 20.2 | 0.78 | 95.0% | 0.0% |
| t16 inner, zero-shot | 120 | 2.5% | 19.5 | 20.7 | 5.4e-07 | 97.4% | 1.4% |
| t16 B inner, adapted | 120 | 12.5% | 13.9 | 20.9 | 2.5e-16 | 80.7% | 11.7% |

- t12 stage A (40 candidates per block): transfer - random mean rank -4.9 (p 1.5e-19): primary met; gate for stage B met; Whisper word errors of the voice 102.1% (sped up 97.5%)
- t12 stage B: primary (adapted inner above its null, p < 0.01) met; vs zero-shot mean rank -9.0 (p 2.3e-26); Whisper word errors 112.0% (sped up 100.4%)
- t16 stage A (40 candidates per block): transfer - random mean rank -15.0 (p 1.4e-20): primary met; gate for stage B met; Whisper word errors of the voice 80.0% (sped up 78.9%)
- t16 stage B: primary (adapted inner above its null, p < 0.01) met; vs zero-shot mean rank -5.7 (p 2.1e-15); Whisper word errors 95.3% (sped up 91.8%)
