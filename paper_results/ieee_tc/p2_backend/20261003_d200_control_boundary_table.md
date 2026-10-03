# D200: instrumented 7B diagnostic (1000-prefix, n=1 run)

Not ordinary Full performance, numerical correctness or SLO qualification.

| Check | Observed |
|---|---:|
| Terminal requests | 1000 / 1000 |
| Native-contract successful | 1000 |
| Physical GPU-s (diagnostic only) | 4452.071693 |
| Control attempts | 4204 |
| Complete-success coverage | True |
| Attempt statuses | {"success_complete": 4204} |
| Partial tails | 0 |

| Business-phase interval | Count | Mean (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| parent_begin__to__parent_send | 4204 | 3.451530 | 1.610799 | 8.980604 |
| parent_send__to__worker_received | 4204 | 5.384395 | 0.295072 | 10.061310 |
| worker_received__to__frontend_begin | 4204 | 0.162640 | 0.137211 | 0.188930 |
| frontend_begin__to__frontend_native_send | 4204 | 0.094644 | 0.093861 | 0.138260 |
| frontend_native_send__to__native_begin | 4204 | 96.328020 | 33.275684 | 408.975018 |
| native_begin__to__native_ready | 4204 | 12.315801 | 10.040065 | 29.037574 |
| native_ready__to__frontend_native_received | 4204 | 24.726790 | 4.460295 | 92.819631 |
| frontend_native_received__to__frontend_ready | 4204 | 1.791394 | 1.512599 | 2.562241 |
| frontend_ready__to__parent_received | 4204 | 35.193664 | 0.716434 | 162.835339 |
| parent_received__to__parent_terminal | 4204 | 0.348078 | 0.236246 | 0.408319 |
| parent_total_ms | 4204 | 179.796957 | 67.515869 | 648.557653 |
| native_thread_cpu_ms | 4204 | 9.520286 | 8.633026 | 15.380119 |

Intervals include observation overhead. Adjacent means are additive only
within one complete-attempt population. Quantiles are not additive; four
concurrent replica calls cannot be summed as request TTFT. Native thread CPU
overlaps native wall time and is shown separately. Business includes drain.

Incomplete/error/cancelled attempts are retained, not imputed zero.
