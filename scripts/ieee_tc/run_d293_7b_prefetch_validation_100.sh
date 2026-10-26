#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d293_20261006/7b_prefetch_validation_100
CONFIG=configs/ieee_tc/20261006_d293_7b_prefetch_validation_100.yaml
TAG=d293_7b_prefetch_validation_100
SCENARIO=tc_ieee_prefetch_validation_100
PREFIX_COUNT=100
case "${1:-d293}" in
  d293) ;;
  d296)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d296_20261007/7b_lifecycle_validation_100
    CONFIG=configs/ieee_tc/20261007_d296_7b_lifecycle_validation_100.yaml
    TAG=d296_7b_lifecycle_validation_100
    ;;
  d297)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d297_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d297_7b_full_w0.yaml
    TAG=d297_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d298)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d298_20261007/7b_capacity_validation_100
    CONFIG=configs/ieee_tc/20261007_d298_7b_capacity_validation_100.yaml
    TAG=d298_7b_capacity_validation_100
    ;;
  d299)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d299_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d299_7b_full_w0.yaml
    TAG=d299_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d300)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d300_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d300_7b_full_w0.yaml
    TAG=d300_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d302)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d302_20261007/7b_frontier_validation_100
    CONFIG=configs/ieee_tc/20261007_d302_7b_frontier_validation_100.yaml
    TAG=d302_7b_frontier_validation_100
    SCENARIO=tc_ieee_frontier_validation_100
    ;;
  d303)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d303_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d303_7b_full_w0.yaml
    TAG=d303_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d308)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d308_20261007/7b_prefill_validation_100
    CONFIG=configs/ieee_tc/20261007_d308_7b_prefill_validation_100.yaml
    TAG=d308_7b_prefill_validation_100
    SCENARIO=tc_ieee_prefill_validation_100
    ;;
  d309)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d309_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d309_7b_full_w0.yaml
    TAG=d309_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d310)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d310_20261007/7b_prefill_validation_100
    CONFIG=configs/ieee_tc/20261007_d310_7b_prefill_validation_100.yaml
    TAG=d310_7b_prefill_validation_100
    SCENARIO=tc_ieee_prefill_validation_100
    ;;
  d311)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d311_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d311_7b_full_w0.yaml
    TAG=d311_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d312)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d312_20261007/7b_prefill_validation_100
    CONFIG=configs/ieee_tc/20261007_d312_7b_prefill_validation_100.yaml
    TAG=d312_7b_prefill_validation_100
    SCENARIO=tc_ieee_prefill_validation_100
    ;;
  d313)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d313_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d313_7b_full_w0.yaml
    TAG=d313_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d314)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d314_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d314_7b_full_w0.yaml
    TAG=d314_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d315)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d315_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d315_7b_full_w0.yaml
    TAG=d315_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d317)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d317_20261007/7b_copy_validation_100
    CONFIG=configs/ieee_tc/20261007_d317_7b_copy_validation_100.yaml
    TAG=d317_7b_copy_validation_100
    SCENARIO=tc_ieee_copy_validation_100
    ;;
  d318)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d318_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d318_7b_full_w0.yaml
    TAG=d318_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d322)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d322_20261007/7b_length_validation_100
    CONFIG=configs/ieee_tc/20261007_d322_7b_length_validation_100.yaml
    TAG=d322_7b_length_validation_100
    SCENARIO=tc_ieee_length_validation_100
    ;;
  d322-attempt2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d322_20261007/7b_length_validation_100_attempt2
    CONFIG=configs/ieee_tc/20261007_d322_7b_length_validation_100_attempt2.yaml
    TAG=d322_7b_length_validation_100_attempt2
    SCENARIO=tc_ieee_length_validation_100
    ;;
  d323)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d323_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d323_7b_full_w0.yaml
    TAG=d323_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d326)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d326_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d326_7b_full_w0.yaml
    TAG=d326_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d328)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d328_20261008/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261008_d328_7b_full_w0_prefix1.yaml
    TAG=d328_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d331)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d331_20261008/7b_planning_validation_100
    CONFIG=configs/ieee_tc/20261008_d331_7b_planning_validation_100.yaml
    TAG=d331_7b_planning_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d331-attempt2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d331_20261008/7b_planning_validation_100_attempt2
    CONFIG=configs/ieee_tc/20261008_d331_7b_planning_validation_100_attempt2.yaml
    TAG=d331_7b_planning_validation_100_attempt2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d332)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d332_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d332_7b_full_w0.yaml
    TAG=d332_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d334)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d334_20261008/7b_native_pending_validation_100
    CONFIG=configs/ieee_tc/20261008_d334_7b_native_pending_validation_100.yaml
    TAG=d334_7b_native_pending_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d335)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d335_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d335_7b_full_w0.yaml
    TAG=d335_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d336)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d336_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d336_7b_full_w0.yaml
    TAG=d336_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d340)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d340_20261008/7b_prebuild_validation_100
    CONFIG=configs/ieee_tc/20261008_d340_7b_prebuild_validation_100.yaml
    TAG=d340_7b_prebuild_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    export FAASLORA_IEEE_QUANTIZATION_PREBUILD_POLICY=on_demand_v1
    export HUMMING_DISABLE_PARALLEL_BUILD=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    ;;
  d341)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d341_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d341_7b_full_w0.yaml
    TAG=d341_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    export FAASLORA_IEEE_QUANTIZATION_PREBUILD_POLICY=on_demand_v1
    export HUMMING_DISABLE_PARALLEL_BUILD=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    ;;
  d345)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d345_20261008/7b_capacity_validation_100
    CONFIG=configs/ieee_tc/20261008_d345_7b_capacity_validation_100.yaml
    TAG=d345_7b_capacity_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d346)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d346_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d346_7b_full_w0.yaml
    TAG=d346_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d349)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d349_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d349_7b_full_w0.yaml
    TAG=d349_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d351)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d351_20261008/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261008_d351_7b_full_w0_prefix1.yaml
    TAG=d351_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d352)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d352_20261008/7b_hit_validation_100
    CONFIG=configs/ieee_tc/20261008_d352_7b_hit_validation_100.yaml
    TAG=d352_7b_hit_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d353)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d353_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d353_7b_full_w0.yaml
    TAG=d353_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d355)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d355_20261008/7b_interleave_validation_100
    CONFIG=configs/ieee_tc/20261008_d355_7b_interleave_validation_100.yaml
    TAG=d355_7b_interleave_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d357)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d357_20261008/7b_interleave_validation_100
    CONFIG=configs/ieee_tc/20261008_d357_7b_interleave_validation_100.yaml
    TAG=d357_7b_interleave_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d358)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d358_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d358_7b_full_w0.yaml
    TAG=d358_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d363)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d363_20261009/7b_file_demand_validation_100
    CONFIG=configs/ieee_tc/20261009_d363_7b_file_demand_validation_100.yaml
    TAG=d363_7b_file_demand_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d364)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d364_20261009/7b_full_w0
    CONFIG=configs/ieee_tc/20261009_d364_7b_full_w0.yaml
    TAG=d364_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d366)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d366_20261009/7b_file_identity_validation_100
    CONFIG=configs/ieee_tc/20261009_d366_7b_file_identity_validation_100.yaml
    TAG=d366_7b_file_identity_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d367)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d367_20261009/7b_full_w0
    CONFIG=configs/ieee_tc/20261009_d367_7b_full_w0.yaml
    TAG=d367_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d368)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d368_20261009/7b_full_w0
    CONFIG=configs/ieee_tc/20261009_d368_7b_full_w0.yaml
    TAG=d368_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d369)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d369_20261010/7b_stale_identity_validation_100
    CONFIG=configs/ieee_tc/20261010_d369_7b_stale_identity_validation_100.yaml
    TAG=d369_7b_stale_identity_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d370)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d370_20261010/7b_full_w0
    CONFIG=configs/ieee_tc/20261010_d370_7b_full_w0.yaml
    TAG=d370_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d371)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d371_20261010/7b_full_w0
    CONFIG=configs/ieee_tc/20261010_d371_7b_full_w0.yaml
    TAG=d371_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d374)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d374_20261010/7b_capacity_revalidation_100
    CONFIG=configs/ieee_tc/20261010_d374_7b_capacity_revalidation_100.yaml
    TAG=d374_7b_capacity_revalidation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d375)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d375_20261010/7b_capacity_revalidation_100
    CONFIG=configs/ieee_tc/20261010_d375_7b_capacity_revalidation_100.yaml
    TAG=d375_7b_capacity_revalidation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d376)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d376_20261010/7b_capacity_revalidation_100
    CONFIG=configs/ieee_tc/20261010_d376_7b_capacity_revalidation_100.yaml
    TAG=d376_7b_capacity_revalidation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d377)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d377_20261010/7b_capacity_revalidation_100
    CONFIG=configs/ieee_tc/20261010_d377_7b_capacity_revalidation_100.yaml
    TAG=d377_7b_capacity_revalidation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d378)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d378_20261010/7b_capacity_revalidation_100
    CONFIG=configs/ieee_tc/20261010_d378_7b_capacity_revalidation_100.yaml
    TAG=d378_7b_capacity_revalidation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d379)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d379_20261010/7b_capacity_revalidation_full_w0
    CONFIG=configs/ieee_tc/20261010_d379_7b_capacity_revalidation_full_w0.yaml
    TAG=d379_7b_capacity_revalidation_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d386)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d386_20261010/7b_capacity_queue4_full_w0
    CONFIG=configs/ieee_tc/20261010_d386_7b_capacity_queue4_full_w0.yaml
    TAG=d386_7b_capacity_queue4_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d450)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/elastic_locality_qualification/d450_20261012/7b_elastic_locality_100
    CONFIG=configs/ieee_tc/20261012_d450_7b_elastic_locality_100.yaml
    TAG=d450_7b_elastic_locality_100
    SCENARIO=tc_ieee_elastic_locality
    PREFIX_COUNT=100
    ;;
  d451)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/elastic_locality_qualification/d451_20261012/7b_elastic_locality_500
    CONFIG=configs/ieee_tc/20261012_d451_7b_elastic_locality_500.yaml
    TAG=d451_7b_elastic_locality_500
    SCENARIO=tc_ieee_elastic_locality
    PREFIX_COUNT=500
    ;;
  d452)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/elastic_locality_qualification/d452_20261012/7b_elastic_locality_1000
    CONFIG=configs/ieee_tc/20261012_d452_7b_elastic_locality_1000.yaml
    TAG=d452_7b_elastic_locality_1000
    SCENARIO=tc_ieee_elastic_locality
    PREFIX_COUNT=1000
    ;;
  d453)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/elastic_locality_qualification/d453_20261012/7b_elastic_locality_w0
    CONFIG=configs/ieee_tc/20261012_d453_7b_elastic_locality_w0.yaml
    TAG=d453_7b_elastic_locality_w0
    SCENARIO=tc_ieee_elastic_locality
    PREFIX_COUNT=
    ;;
  d456)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d456_20261012/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261012_d456_7b_full_w0_prefix1.yaml
    TAG=d456_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d457)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d457_20261012/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261012_d457_7b_full_w0_prefix1.yaml
    TAG=d457_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d458)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d458_20261012/7b_full_w0
    CONFIG=configs/ieee_tc/20261012_d458_7b_full_w0.yaml
    TAG=d458_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d464)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d464_20261012/7b_host_queue_validation_100
    CONFIG=configs/ieee_tc/20261012_d464_7b_host_queue_validation_100.yaml
    TAG=d464_7b_host_queue_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d465)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d465_20261012/7b_full_w0
    CONFIG=configs/ieee_tc/20261012_d465_7b_full_w0.yaml
    TAG=d465_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d470)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d470_20261012/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261012_d470_7b_cpu_frontend_validation_100.yaml
    TAG=d470_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d471)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d471_20261012/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261012_d471_7b_cpu_frontend_validation_100.yaml
    TAG=d471_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d472)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d472_20261012/7b_full_w0
    CONFIG=configs/ieee_tc/20261012_d472_7b_full_w0.yaml
    TAG=d472_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d475)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d475_20261012/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261012_d475_7b_cpu_frontend_validation_100.yaml
    TAG=d475_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d476)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d476_20261012/7b_full_w0
    CONFIG=configs/ieee_tc/20261012_d476_7b_full_w0.yaml
    TAG=d476_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d512)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d512_20261014/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261014_d512_7b_full_w0_prefix1.yaml
    TAG=d512_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d515)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d515_20261014/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261014_d515_7b_cpu_frontend_validation_100.yaml
    TAG=d515_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d516)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d516_20261014/7b_full_w0
    CONFIG=configs/ieee_tc/20261014_d516_7b_full_w0.yaml
    TAG=d516_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d519)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d519_20261014/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261014_d519_7b_full_w0_prefix1.yaml
    TAG=d519_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d523)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d523_20261014/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261014_d523_7b_cpu_frontend_validation_100.yaml
    TAG=d523_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d527)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d527_20261014/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261014_d527_7b_cpu_frontend_validation_100.yaml
    TAG=d527_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d528)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d528_20261014/7b_cpu_frontend_validation_100
    CONFIG=configs/ieee_tc/20261014_d528_7b_cpu_frontend_validation_100.yaml
    TAG=d528_7b_cpu_frontend_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d529)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d529_20261014/7b_full_w0
    CONFIG=configs/ieee_tc/20261014_d529_7b_full_w0.yaml
    TAG=d529_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d533)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d533_20261015/7b_full_w1
    CONFIG=configs/ieee_tc/20261015_d533_7b_full_w1.yaml
    TAG=d533_7b_full_w1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d550)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d550_20261015/7b_initial_control_validation_100
    CONFIG=configs/ieee_tc/20261015_d550_7b_initial_control_validation_100.yaml
    TAG=d550_7b_initial_control_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d551)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d551_20261015/7b_initial_control_full_w1
    CONFIG=configs/ieee_tc/20261015_d551_7b_initial_control_full_w1.yaml
    TAG=d551_7b_initial_control_full_w1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d554)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d554_20261015/7b_initial_control_full_w2
    CONFIG=configs/ieee_tc/20261015_d554_7b_initial_control_full_w2.yaml
    TAG=d554_7b_initial_control_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d597)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d597_20261017/7b_parallel_pressure_validation_100
    CONFIG=configs/ieee_tc/20261017_d597_7b_parallel_pressure_validation_100.yaml
    TAG=d597_7b_parallel_pressure_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d598)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d598_20261017/7b_parallel_pressure_full_w2
    CONFIG=configs/ieee_tc/20261017_d598_7b_parallel_pressure_full_w2.yaml
    TAG=d598_7b_parallel_pressure_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d603)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d603_20261017/7b_controller_full_w2
    CONFIG=configs/ieee_tc/20261017_d603_7b_controller_full_w2.yaml
    TAG=d603_7b_controller_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d607)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d607_20261017/7b_canonical_files_full_w2
    CONFIG=configs/ieee_tc/20261017_d607_7b_canonical_files_full_w2.yaml
    TAG=d607_7b_canonical_files_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d611)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d611_20261017/7b_inventory_extent_full_w2
    CONFIG=configs/ieee_tc/20261017_d611_7b_inventory_extent_full_w2.yaml
    TAG=d611_7b_inventory_extent_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d615)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d615_20261017/7b_three_replica_full_w2
    CONFIG=configs/ieee_tc/20261017_d615_7b_three_replica_full_w2.yaml
    TAG=d615_7b_three_replica_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d617)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d617_20261017/7b_three_replica_full_w1
    CONFIG=configs/ieee_tc/20261017_d617_7b_three_replica_full_w1.yaml
    TAG=d617_7b_three_replica_full_w1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d619)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d619_20261017/7b_device_domain_full_w1
    CONFIG=configs/ieee_tc/20261017_d619_7b_device_domain_full_w1.yaml
    TAG=d619_7b_device_domain_full_w1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d621)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d621_20261017/7b_device_domain_full_w2
    CONFIG=configs/ieee_tc/20261017_d621_7b_device_domain_full_w2.yaml
    TAG=d621_7b_device_domain_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d624)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d624_20261017/7b_device_domain_full_w0
    CONFIG=configs/ieee_tc/20261017_d624_7b_device_domain_full_w0.yaml
    TAG=d624_7b_device_domain_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d629)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d629_20261017/7b_private_verify_full_w2
    CONFIG=configs/ieee_tc/20261017_d629_7b_private_verify_full_w2.yaml
    TAG=d629_7b_private_verify_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d632)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d632_20261017/7b_min2_full_w2
    CONFIG=configs/ieee_tc/20261017_d632_7b_min2_full_w2.yaml
    TAG=d632_7b_min2_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d633)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d633_20261017/7b_min2_full_w1
    CONFIG=configs/ieee_tc/20261017_d633_7b_min2_full_w1.yaml
    TAG=d633_7b_min2_full_w1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d634)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d634_20261017/7b_min2_full_w0
    CONFIG=configs/ieee_tc/20261017_d634_7b_min2_full_w0.yaml
    TAG=d634_7b_min2_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d636)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d636_20261017/7b_min2_full_w0
    CONFIG=configs/ieee_tc/20261017_d636_7b_min2_full_w0.yaml
    TAG=d636_7b_min2_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d659)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d659_20261018/3b_full_smoke100
    CONFIG=configs/ieee_tc/20261018_d659_3b_full_smoke100.yaml
    TAG=d659_3b_full_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d660)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d660_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d660_3b_full_w0.yaml
    TAG=d660_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d664)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d664_20261018/3b_full_smoke100
    CONFIG=configs/ieee_tc/20261018_d664_3b_full_smoke100.yaml
    TAG=d664_3b_full_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d665)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d665_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d665_3b_full_w0.yaml
    TAG=d665_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d668)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d668_20261018/3b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261018_d668_3b_full_w0_prefix1.yaml
    TAG=d668_3b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d672)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d672_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d672_3b_full_w0.yaml
    TAG=d672_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d676)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d676_20261018/3b_full_smoke100
    CONFIG=configs/ieee_tc/20261018_d676_3b_full_smoke100.yaml
    TAG=d676_3b_full_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d677)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d677_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d677_3b_full_w0.yaml
    TAG=d677_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d681)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d681_20261018/3b_full_smoke100
    CONFIG=configs/ieee_tc/20261018_d681_3b_full_smoke100.yaml
    TAG=d681_3b_full_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d694)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d694_20261018/3b_full_smoke100
    CONFIG=configs/ieee_tc/20261018_d694_3b_full_smoke100.yaml
    TAG=d694_3b_full_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d696)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d696_20261018/3b_full_smoke100
    CONFIG=configs/ieee_tc/20261018_d696_3b_singleflight_smoke100.yaml
    TAG=d696_3b_singleflight_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d702)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d702_20261019/3b_profile_smoke100
    CONFIG=configs/ieee_tc/20261019_d702_3b_profile_smoke100.yaml
    TAG=d702_3b_profile_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d718)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d718_20261022/3b_identity_smoke100
    CONFIG=configs/ieee_tc/20261022_d718_3b_identity_smoke100.yaml
    TAG=d718_3b_identity_smoke100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d843)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d843_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d843_3b_max3_full_w2.yaml
    TAG=d843_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d854)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d854_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d854_3b_max3_full_w2.yaml
    TAG=d854_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d861)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d861_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d861_3b_max3_full_w2.yaml
    TAG=d861_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d863_attempt2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d863_20261026_attempt2/3b_native_prefix1000
    CONFIG=configs/ieee_tc/20261026_d863_attempt2_3b_native_prefix1000.yaml
    TAG=d863_attempt2_3b_native_prefix1000
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d863)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d863_20261026/3b_native_prefix1000
    CONFIG=configs/ieee_tc/20261026_d863_3b_native_prefix1000.yaml
    TAG=d863_3b_native_prefix1000
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d858)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d858_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d858_3b_max3_full_w2.yaml
    TAG=d858_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d870)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d870_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d870_3b_max3_full_w2.yaml
    TAG=d870_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d883)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d883_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d883_3b_max3_full_w2.yaml
    TAG=d883_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d886)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d886_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d886_3b_max3_full_w2.yaml
    TAG=d886_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d876)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d876_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d876_3b_max3_full_w2.yaml
    TAG=d876_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d880)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d880_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d880_3b_max3_full_w2.yaml
    TAG=d880_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d866)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d866_20261026/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261026_d866_3b_max3_full_w2.yaml
    TAG=d866_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d846)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d846_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d846_3b_max3_full_w2.yaml
    TAG=d846_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d840)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d840_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d840_3b_max3_full_w2.yaml
    TAG=d840_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d835)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d835_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d835_3b_max3_full_w2.yaml
    TAG=d835_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d832)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d832_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d832_3b_max3_full_w2.yaml
    TAG=d832_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d829)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d829_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d829_3b_max3_full_w2.yaml
    TAG=d829_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d826)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d826_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d826_3b_max3_full_w2.yaml
    TAG=d826_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d822)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d822_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d822_3b_max3_full_w2.yaml
    TAG=d822_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d817)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d817_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d817_3b_max3_full_w2.yaml
    TAG=d817_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d814)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d814_20261025/3b_max3_full_w2
    CONFIG=configs/ieee_tc/20261025_d814_3b_max3_full_w2.yaml
    TAG=d814_3b_max3_full_w2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d813)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d813_20261025/3b_max3_full_w1
    CONFIG=configs/ieee_tc/20261025_d813_3b_max3_full_w1.yaml
    TAG=d813_3b_max3_full_w1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d812)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d812_20261024/3b_max3_full_w0
    CONFIG=configs/ieee_tc/20261024_d812_3b_max3_full_w0.yaml
    TAG=d812_3b_max3_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d752)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d752_20261023/3b_dense_alias_validation_full_w0
    CONFIG=configs/ieee_tc/20261023_d752_3b_dense_alias_validation_full_w0.yaml
    TAG=d752_3b_dense_alias_validation_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d749)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d749_20261023/3b_fresh_host_descriptor_full_w0
    CONFIG=configs/ieee_tc/20261023_d749_3b_fresh_host_descriptor_full_w0.yaml
    TAG=d749_3b_fresh_host_descriptor_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d745)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d745_20261023/3b_observe_after_turn_full_w0
    CONFIG=configs/ieee_tc/20261023_d745_3b_observe_after_turn_full_w0.yaml
    TAG=d745_3b_observe_after_turn_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d742)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d742_20261023/3b_execution_view_full_w0
    CONFIG=configs/ieee_tc/20261023_d742_3b_execution_view_full_w0.yaml
    TAG=d742_3b_execution_view_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d738)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d738_20261023/3b_planning_wire_full_w0
    CONFIG=configs/ieee_tc/20261023_d738_3b_planning_wire_full_w0.yaml
    TAG=d738_3b_planning_wire_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d736)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d736_20261022/3b_validated_publication_full_w0
    CONFIG=configs/ieee_tc/20261022_d736_3b_validated_publication_full_w0.yaml
    TAG=d736_3b_validated_publication_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d734)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d734_20261022/3b_controller_prefix1000
    CONFIG=configs/ieee_tc/20261022_d734_3b_controller_prefix1000.yaml
    TAG=d734_3b_controller_prefix1000
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d732)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d732_20261022/3b_round_file_full_w0
    CONFIG=configs/ieee_tc/20261022_d732_3b_round_file_full_w0.yaml
    TAG=d732_3b_round_file_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d729)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d729_20261022/3b_bulk_loader_full_w0
    CONFIG=configs/ieee_tc/20261022_d729_3b_bulk_loader_full_w0.yaml
    TAG=d729_3b_bulk_loader_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d726)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d726_20261022/3b_graph_full_w0
    CONFIG=configs/ieee_tc/20261022_d726_3b_graph_full_w0.yaml
    TAG=d726_3b_graph_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d719)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d719_20261022/3b_identity_full_w0
    CONFIG=configs/ieee_tc/20261022_d719_3b_identity_full_w0.yaml
    TAG=d719_3b_identity_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d713)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d713_20261022/3b_control_prefix1000
    CONFIG=configs/ieee_tc/20261022_d713_3b_control_prefix1000.yaml
    TAG=d713_3b_control_prefix1000
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d703)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d703_20261019/3b_full_w0
    CONFIG=configs/ieee_tc/20261019_d703_3b_full_w0_d701profile.yaml
    TAG=d703_3b_full_w0_d701profile
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d697)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d697_20261019/3b_full_w0
    CONFIG=configs/ieee_tc/20261019_d697_3b_full_w0.yaml
    TAG=d697_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d695)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d695_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d695_3b_full_w0.yaml
    TAG=d695_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d682)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d682_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d682_3b_full_w0.yaml
    TAG=d682_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d688)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d688_20261018/3b_full_w0
    CONFIG=configs/ieee_tc/20261018_d688_3b_full_w0.yaml
    TAG=d688_3b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d684)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d684_20261018/3b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261018_d684_3b_full_w0_prefix1.yaml
    TAG=d684_3b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d723)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d723_20261022/3b_step_components100
    CONFIG=configs/ieee_tc/20261022_d723_3b_step_components100.yaml
    TAG=d723_3b_step_components100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  *) echo 'Unknown frozen campaign identity; no arbitrary output override' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || exit 2
cd "$ROOT"
[[ ! -e "$RAW" ]] || { echo "Refusing existing run: $RAW" >&2; exit 2; }
mkdir -p "$(dirname "$RAW")"
mkdir "$RAW"
set -o noclobber
exec >"$RAW/launcher.log" 2>&1
export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
export MAX_JOBS=2
IFS= read -r PRIME_REMOTE_TOKEN < /home/qhq/.config/primelora-tc-d75/artifact.token
export PRIME_REMOTE_TOKEN
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18081
export FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE=prepublished_gzip_v1
export FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
unset FAASLORA_IEEE_HOST_INVENTORY_MODE
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
export VLLM_CACHE_ROOT=/home/qhq/.cache/vllm
export FAASLORA_PYTHON=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
if [[ "${1:-d293}" == d328 || "${1:-d293}" == d351 || "${1:-d293}" == d456 || "${1:-d293}" == d512 || "${1:-d293}" == d519 || "${1:-d293}" == d603 || "${1:-d293}" == d668 || "${1:-d293}" == d684 || "${1:-d293}" == d723 || "${1:-d293}" == d734 || "${1:-d293}" == d843 || "${1:-d293}" == d863 || "${1:-d293}" == d863_attempt2 ]]; then
  # Observer is enabled by this wrapper only AFTER the normal resource gate.
  export FAASLORA_PYTHON="$ROOT/scripts/ieee_tc/profile_d328_controller_python.sh"
fi
export FAASLORA_TC_QUALIFICATION=1
export FAASLORA_TC_EXTERNAL_REPLAY=1
export FAASLORA_TC_REPLAY_PROFILE=W0
unset FAASLORA_TC_ADAPTER_MAP FAASLORA_TC_ADAPTER_MAP_SHA256
if [[ "${1:-d293}" == d554 || "${1:-d293}" == d598 || "${1:-d293}" == d603 || "${1:-d293}" == d607 || "${1:-d293}" == d611 || "${1:-d293}" == d615 || "${1:-d293}" == d621 || "${1:-d293}" == d629 || "${1:-d293}" == d632 ]]; then
  export FAASLORA_TC_REPLAY_PROFILE=W2
  export FAASLORA_TC_ADAPTER_MAP="$ROOT/paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json"
  export FAASLORA_TC_ADAPTER_MAP_SHA256=48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
fi
if [[ "${1:-d293}" == d533 || "${1:-d293}" == d550 || "${1:-d293}" == d551 || "${1:-d293}" == d597 || "${1:-d293}" == d617 || "${1:-d293}" == d619 || "${1:-d293}" == d633 ]]; then
  export FAASLORA_TC_REPLAY_PROFILE=W1
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
fi
if [[ "${1:-d293}" == d624 ]]; then
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
fi
if [[ "${1:-d293}" == d634 ]]; then
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
fi
if [[ "${1:-d293}" == d636 ]]; then
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
fi
if [[ "${1:-d293}" == d659 || "${1:-d293}" == d660 || "${1:-d293}" == d664 || "${1:-d293}" == d665 || "${1:-d293}" == d668 || "${1:-d293}" == d672 || "${1:-d293}" == d676 || "${1:-d293}" == d677 || "${1:-d293}" == d681 || "${1:-d293}" == d694 || "${1:-d293}" == d696 || "${1:-d293}" == d697 || "${1:-d293}" == d695 || "${1:-d293}" == d702 || "${1:-d293}" == d703 || "${1:-d293}" == d713 || "${1:-d293}" == d718 || "${1:-d293}" == d719 || "${1:-d293}" == d682 || "${1:-d293}" == d688 || "${1:-d293}" == d684 || "${1:-d293}" == d723 || "${1:-d293}" == d726 || "${1:-d293}" == d729 || "${1:-d293}" == d732 || "${1:-d293}" == d734 || "${1:-d293}" == d736 || "${1:-d293}" == d738 || "${1:-d293}" == d742 || "${1:-d293}" == d745 || "${1:-d293}" == d749 || "${1:-d293}" == d752 ]]; then
  export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
fi
if [[ -n "$PREFIX_COUNT" ]]; then
  export FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT="$PREFIX_COUNT"
else
  unset FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT
fi
export FAASLORA_TC_LAUNCH_OUTPUT="$RAW/launch.json"
export FAASLORA_TC_PREDICTED_GROWTH_GIB=32
export FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
export FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json
export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
if [[ "${1:-d293}" == d659 || "${1:-d293}" == d660 || "${1:-d293}" == d664 || "${1:-d293}" == d665 || "${1:-d293}" == d668 || "${1:-d293}" == d672 || "${1:-d293}" == d676 || "${1:-d293}" == d677 || "${1:-d293}" == d681 || "${1:-d293}" == d694 || "${1:-d293}" == d696 || "${1:-d293}" == d697 || "${1:-d293}" == d695 || "${1:-d293}" == d702 || "${1:-d293}" == d703 || "${1:-d293}" == d713 || "${1:-d293}" == d718 || "${1:-d293}" == d719 || "${1:-d293}" == d682 || "${1:-d293}" == d688 || "${1:-d293}" == d684 || "${1:-d293}" == d723 || "${1:-d293}" == d726 || "${1:-d293}" == d729 || "${1:-d293}" == d732 || "${1:-d293}" == d734 || "${1:-d293}" == d736 || "${1:-d293}" == d738 || "${1:-d293}" == d742 || "${1:-d293}" == d745 || "${1:-d293}" == d749 || "${1:-d293}" == d752 ]]; then
  export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
  export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
fi
if [[ "${1:-d293}" == d749 ]]; then
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
fi
if [[ "${1:-d293}" == d752 ]]; then
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
fi
if [[ "${1:-d293}" == d812 ]]; then
  export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
  export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
  export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
fi
if [[ "${1:-d293}" == d814 || "${1:-d293}" == d817 || "${1:-d293}" == d822 || "${1:-d293}" == d826 || "${1:-d293}" == d829 || "${1:-d293}" == d832 || "${1:-d293}" == d835 || "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d861 || "${1:-d293}" == d858 || "${1:-d293}" == d870 || "${1:-d293}" == d883 || "${1:-d293}" == d886 || "${1:-d293}" == d876 || "${1:-d293}" == d880 || "${1:-d293}" == d866 || "${1:-d293}" == d854 || "${1:-d293}" == d846 || "${1:-d293}" == d863 || "${1:-d293}" == d863_attempt2 ]]; then
  export FAASLORA_TC_REPLAY_PROFILE=W2
  export FAASLORA_TC_ADAPTER_MAP="$ROOT/paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json"
  export FAASLORA_TC_ADAPTER_MAP_SHA256=28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23
  export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
  export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
  export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
fi
if [[ "${1:-d293}" == d813 ]]; then
  export FAASLORA_TC_REPLAY_PROFILE=W1
  export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
  export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
  export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
fi
if [[ "${1:-d293}" == d451 ]]; then
  # Reuse D433/D443's original source and exact derived functional view.
  export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
  export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
  export FAASLORA_TC_REPLAY_PROFILE=ADAPTER_COVERAGE_V1
  export FAASLORA_TC_COVERAGE_POOL_INDEX="$ROOT/paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json"
  export FAASLORA_TC_COVERAGE_POOL_INDEX_SHA256=e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c
fi
if [[ "${1:-d293}" == d840 || "${1:-d293}" == d843 || "${1:-d293}" == d861 || "${1:-d293}" == d858 || "${1:-d293}" == d870 || "${1:-d293}" == d883 || "${1:-d293}" == d886 || "${1:-d293}" == d876 || "${1:-d293}" == d880 || "${1:-d293}" == d866 || "${1:-d293}" == d854 || "${1:-d293}" == d846 || "${1:-d293}" == d863 || "${1:-d293}" == d863_attempt2 ]]; then
  export FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal
  export PYTHONPATH="/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d838_20261025/dependency:$ROOT"
fi
if [[ "${1:-d293}" == d866 ]]; then
  export FAASLORA_IEEE_HOST_INVENTORY_MODE=graph
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d864_20261026/attempt3/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=30455d8f678483eb1adf3f7e594ffb6bef90a7aad462fbc03efbd98b969edf80
fi
export FAASLORA_TRACE_ROLE=exploratory
export FAASLORA_FORMAL_RUN=0
export FAASLORA_RESULTS_TAG="$TAG"
export FAASLORA_WORKER_START_TIMEOUT_S=600
export FAASLORA_KEEP_DEDICATED_WORKER_LOGS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
exec bash scripts/run_all_experiments_user_scope.sh \
  --config "$CONFIG" \
  --scenario "$SCENARIO" --full-stack
