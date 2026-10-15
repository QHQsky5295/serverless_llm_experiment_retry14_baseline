# IEEE TC execution status — D537 ServerlessLLM W1 qualified, prelaunch backup next

Updated 2026-10-15. Current authoritative handoff; not a new experimental plan.
Prior status preserved BYTE-FOR-BYTE in EXECUTION_HISTORY_D520_THROUGH_D536_20261015.md
SHA9aa084e924696efd2ef84879837707b6626a9bdc43282d2f1127d19c002cba05.
Earlier EXECUTION_HISTORY_THROUGH_D519.md SHA
e0ed5415b779effec429b48ea716abde2e3e9b1a379d5abbc11073cdca2f3a99.
Historical running/prepared notes in archives are superseded here.

## Immediate state / next action

D533 TERMINAL SUCCESS and sealed; read D533_FULL_W1_RESULT.md.
D537 ServerlessLLM W1 prepared/qualified, NOT YET STARTED. Read
D537_SERVERLESSLLM_W1.md. RAW originalresults/ieee_tc/serverless_qualification/d537_20261015.
Reuse D414 native0102/store/RR config, only explicit trace_w1_v1/replay_profileW1
and new paths. No serving code or native policy changes. All114 CPU tests closed:
112 native-env passes;2 analysis-only methods had6 ijson-missing subcase errors,
then only those2 passed in existing analysisenv. No dependency installed.
Firstoverstrict AST assertion failed beforeviewcreation; exactsharedhook review
retained. All3aux scopes closed/recursiveempty/pathgone/events0/swap0;
peaks137654272/344240128/149884928B. No GPU run/overlay/remote start yet.
NEXT explicit prelaunch backup/push, freshsafetygate, ownloaderinstall, remote
health, one4000W1 run. Do not repeat qualification or D533/D536/W0.

D533 actual MAINfc4161a1088efe1b49d853dbbc8effe9a525c431,BASE
bde215460acb9e850efa479a89bda6fc39a2bd7d; both pushed before launch.
RAW /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d533_20261015.
Curated paper_results/ieee_tc/main_comparison/d533_full_w1/result1:
audit.json,cross_contract.json,manifest.json,evidence.tar.gz84895B.
Curation DONE, actualresultscope recursivelyempty/closed/pathgone,peak26853376B,
events0/swap0. Testedpublication COMPLETE:bb5f355a7c274ac027c9e35f7eb3375bdaf9a06e
PUSHED/remoteequal,9explicitfiles. Archive/SHA/protected147/userdirt/diff/secrets
checksPASS. Actualpublicationscope recursiveempty/closed/pathgone,peak27127808B,
events0/swap0. RAWpublication_receipt/closure retain identity. Do not repeat backup.
Only status and untoucheduser manifest were tracked dirty after D533 backup.
D531103tests,D53416tests,D535build reused, not repeated. No collection/analysis rerun.

4000nativebindings/458224tokens/0failures,n_correctnull. Joint3749=93.725%,
TTFT3788,TPOT3950. P95TTFT4473.332113120705ms,meanTTFT1625.2706951366854,
meanTPOT35.24462606951364,P95TPOT58.25185352918126,meanE2E5649.552373536164.
U8145.351609454025,R2.036337902363506/binding-validrequest. NOT95%feasible.
Timing/E2E identities<=1ms. GPU windows pre35.46371716680005,
arrival8076.458608148387,drain3.6708148929756135,cleanup29.758469245862216.
Initial1/natural5 ready activations; initial trigger->ready90.49955091997981s.
Dispatch gpu2736/host321/nvme809/remote134. Remote132fetch/131480060wireB/
packing0,bilateraljoin/hashpass. Phasejoint409,387,499,488,489,491,491,495:
204/251violations in first1000, no permission to discard or causalclaim allJIT.
Servicepeak20661305344B,allhigh/max/OOM/kill/swap0; allGPU/service/launcher
released,actualrecursiveempty/pathgone. All3ownedremoteunitsinactive/portsabsent.
Analysisclosedpeak1068179456B; successful inputjoin closedpeak687910912B.
First join failed expecting baseline compiler receipt in log; actual is
worker_receipts JSON. Failure source/receipt retained, no inference rerun.
Intermediate cache-path regex overescaped; finalcuration restored from literal
retained log lines; raw intermediate unchanged, no input/metric changes.

## W1 reference / common software

D536 vLLM sealed/pushed3090d9130a78b4e67d3ed7e929e71ed15766caab; read
D536_VLLM_W1_RESULT.md.4000native/458224tokens, joint2907=72.675%,
U9008.761400188087,P95TTFT9777.026765514165ms,alljobsclosed. No rerun.
Matched4000W1 input join DONE. D533 vsD536 U9.584112%lower,
P95TTFT54.246498%lower,joint+21.05pp; TPOT worse. BOTHbelow95%.
W1 Uref/Qref NOT formally frozen. Pre-D533reference_candidate:
U9008.761400188087,Q9777.026765514165,B6756.571050141065;
D533U/B1.205545172, noG2feasibility/optimum.

D534sharedtoolchain+D535genericsampling DONE. Config
configs/ieee_tc/native_toolchain_cuda130_v1.json SHA
328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c.
Native /home/qhq/.venvs/primelora_vllm0300_tc_20260925; CUDA_HOME
/usr/local/cuda-13.0,CC=/usr/bin/gcc,CXX=/usr/bin/c++,NVCCcuda13.
sampling.soSHA0c8303aa574025abf1998836a42dde6092a8cd1701fee75b1dc610429432e76d.
Do not rebuild/retestD531/D534/D535 without affected changes.
D532oldcompiler-cacheinterference retained, not new reference or subtracted.
D536 initialprefix6d1961d6a9 compile9.01s/total31.58s/init56.71s.
D533initialprefix3a5577e355 absentprelaunchmetadata,compile8.81s/total30.50s/
init56.49s. Later5reuseAOT936104b991e9ff22cb3c4b7698d45eee90f7200228bc4570e9e65966dab81fc6.
FirstLoRAJIT included; no extra prewarm/subtraction. Wholelowerkernelprestate
notproved;14slot/rank16 vs4slot/rank64 remains explicit. No singlemechanismcause.

## Authority / metrics / retained history

Before work/aftercompaction/beforeexperiment read FULL
/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this status.
PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83.
Read STORAGE_PRIORITY_20261013,BASELINE_COMPLETION_ORDER_20261012,
BASELINE_EXECUTION_RULES_20261011,BASELINE_SCOPE_20261014,
SLO_BUDGET_RATIONALE_20261014.md. D483cleanup DONE; no broadrepeat.
HydraServeD510isolatedrootlessPodman/full4000 DONE. Sharedk3s is unrelated:
do not alter cluster/permissions/credentials. Latest reply does not require redo.

Publishedroster PrimeLoRA,vLLM,S-LoRA,ServerlessLLM,HydraServe,Loquetier;
dLoRAonly3BW1/W2. ElasticLocality ONLYcompositeablation,notbaseline.
Baseline necessaryadaptation/abnormal-resultdefectreview only,no sweeps.
Beforecomparison/configselectionread FULL METRIC_PROTOCOL_FROZEN_V1.md SHA
5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22 and
METRIC_PROTOCOL_SINGLE_RUN_V2.md SHA193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
G1allcorrect/native+joint>=95 thenminphysicalU;G2same+U<=.75Uref,W1/W2P95
andobservedSLOdifference. CEsupplementary,independentnumericalunknownnotproof.
Singleperformance/key,noCI/repeatmeans/winnerfilter.95/.75projectpointsnotnorms;
90/95/99,.5/.75/1/multiplier sensitivity retained. Fullnames; Chinese updates.
Nofiguresuntilneartarget;no newweights/traces/pools/13B/SGLang/LaTeXedits.
Nineformulas and cross-scenariofrozenmodelconfig. Eachoptimizationhistory+
primaryofficialsources+falsifiablehypothesis+minimalvalidation+fullreplay.

D476retainedW0 joint97.425%,U10778.448802226456,R2.694612200556614,
P95TTFT2315.363336ms. D513publishedtableSHA
044c04d6bb5b894761628146a65de248a53436cdd4ba23346837eccb3473b17b.
W0(system,joint,R,P95ms):ServerlessLLMD414 .98125/4.022440686/1052.246361;
vLLMD423 .99475/4.028844291/1200.506860;S-LoRAD435 .94425/3.789269468/17699.067366;
LoquetierD445 .0435/4.249595809/555110.032229;HydraServeD510 .68725/2.415538153/8851.575365.
LastthreeNONFEASIBLE. HydraServe34workersallpp1,no pipeline/consolidationproof.
ElasticLocalityD453 .95775/2.473178968/2420.607829 ablationonly.
D516REJECTED .94775/U11011.303223;D529metadataREJECTED .955/U11508.173072/
P952893.673578; policyOFF. D530stage/controlreviewDONE,no repeat.
D519componentjournalpartial64MiBcap retained,notwholeprocessproof.

D394SLOSHA2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3:
input<=616/>616 TTFT2995.348713684507/4633.345350589479ms,
TPOT59.74409694902162/85.24698166015581ms. W0Uref16115.377162149176,
B12086.532871611882,Q1200.5068596918136. W0notW1/W2/G2closure.
Source trace/W1view identities in D533 resultdoc and audit.

## Safety / protection / publishing

Oneheavyjob; no tests/build/hashscan/compression/backup/remotechangesduringinference.
Servicehigh72/max80GiB,swap2,CPU4-23,28-47;auxhigh3/max4GiB,swap0,CPU2,3,26,27.
EmptyhoststartMemAvailable>=102GiB,remaininggrowthotherwise;16GiBstop.
Diskstart>=max150GiB,100GiB+1.5growth;stop100GiB. Freshgatebeforenextrun.
Latestpostbackup disk162879889408B,MemAvailable109282876kB; no new run launched.
Actualworkerscontained; no blindOOMretry/globalGPUreset/unknowncleanup.
ExactInvocation/inode/recursivecgroup.procs/pathgoneclosure,UID1001.
D533/D536/D532privatefetchcaches retained; reclaimonlyafterreferences/openfiles/
retainedcontentsourceaudit,notbydirname. Uniqueassets/failures/userdirtprotected.

MAINretry14_continuous_queue_v2/faaslora_origin;BASEmain/origin.
Protected147 originalresults/ieee_tc/slora_qualification/d435_20261011/protected_before.json
SHA79df0c64c0d88236bc20b439d735f8dcab113f06582421151742aa63a5a2be54.
NeverstageMAINconfigs/generated/lora_manifest_1000.json SHA
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df.
BASEscripts/replay_openai_trace.py SHA04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
scripts/run_serverlessllm_relayserve_continuation.sh SHA337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
Explicitpaths/tests/SHA/archive/diff/secretsbeforepush,noforce.
SSHprimelora-artifact-174 lab14@10.199.227.174:8122strictknownhost;
data192.168.4.174ports18080/18081 currentlyINACTIVE. Privatetoken
/home/qhq/.config/primelora-tc-d75/artifact.token NEVERprint/stage.
D78/D80readonlygzip1581215261B reuse,norepublish/newpool/requestpacking.
Latestactual1Gbps/full;old100Mbpsdistinct,causeunknown.

## Full remaining scope — ACTIVE

D537 newrun7b_repaired/sourceview/http/start/restore/close helpers exist.
Sourceview4measuredfileSHAs equalD414, native configASTunchanged.
W1viewSHA103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3;
4000/132IDs/458224tokens, allsourceassetsreused. Existinganalyzerfull4000_w1.
LoaderpreflightPASS only; startmustinstallownoverlay7b_repaired receipt/backups.
Restoreonlyafteractualrelease. Do not executeD414oldpaths or oldrestore receipt.

7BW1/W2/G2/commonrefs/allpublishedsystems;PrimeLoRAevidence-drivenoptimization;
3Boptimization/allbaselines;dLoRA3BB6;A1–A5confirmedstate/elasticity/handoff/
hierarchy/admission/firstservice;S1–S13tier/mismatch/contention/load/GPU/network/
workload/KV/control/backend/EOS/design;offlinecost/CE/Pareto;B1/B2/C1/C2/C3/
formula/novelty/reviewdocs;lateracademicfigures/V2delivery. No goalcompletion.
