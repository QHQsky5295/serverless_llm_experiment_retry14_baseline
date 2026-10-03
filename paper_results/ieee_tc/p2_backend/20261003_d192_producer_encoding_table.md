# D192 producer encoding component table

Three alternating CPU component rounds, not independent serving runs. Guarded includes all validation.
Rejected candidate: no serving change or corresponding full replay. Negative reductions are regressions.

| 快照 | 消息形态 | 原 bytes | 原型 bytes | 原 ms | 校验＋原型 ms | 下降比例 |
|---:|---|---:|---:|---:|---:|---:|
| -1 | existing_small_primitive_fixture | 305 | 262 | 0.008337 | 0.006725 | 19.333% |
| 0 | current_identity_projection | 3259 | 3088 | 0.032324 | 0.029949 | 7.346% |
| 0 | current_request_projection | 3578 | 3393 | 0.034334 | 0.034153 | 0.527% |
| 0 | current_routing_projection | 5249 | 4994 | 0.045552 | 0.046059 | -1.114% |
| 0 | historical_full_native_graph | 1518630 | 1388829 | 23.211254 | 33.187543 | -42.980% |
| 1 | current_identity_projection | 4554 | 4315 | 0.044102 | 0.042241 | 4.220% |
| 1 | current_request_projection | 4873 | 4620 | 0.047435 | 0.046036 | 2.950% |
| 1 | current_routing_projection | 7496 | 7133 | 0.065522 | 0.064365 | 1.766% |
| 1 | historical_full_native_graph | 2215094 | 2025713 | 26.546090 | 44.937627 | -69.282% |
| 2 | current_identity_projection | 7143 | 6768 | 0.056782 | 0.056885 | -0.182% |
| 2 | current_request_projection | 7462 | 7073 | 0.058807 | 0.059338 | -0.903% |
| 2 | current_routing_projection | 11989 | 11410 | 0.089873 | 0.091591 | -1.911% |
| 2 | historical_full_native_graph | 3609914 | 3301361 | 40.924184 | 71.698431 | -75.198% |
| 3 | current_identity_projection | 8433 | 7990 | 0.065369 | 0.062185 | 4.870% |
| 3 | current_request_projection | 8750 | 8293 | 0.067025 | 0.066686 | 0.507% |
| 3 | current_routing_projection | 14229 | 13542 | 0.104243 | 0.104764 | -0.500% |
| 3 | historical_full_native_graph | 4304329 | 3936194 | 47.799010 | 82.341437 | -72.266% |
