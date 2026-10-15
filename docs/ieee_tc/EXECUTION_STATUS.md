# IEEE TC execution status — D540 S-LoRA W1 prelaunch

Updated 2026-10-15. Current authoritative handoff; not a new experimental plan.
Prior status preserved BYTE-FOR-BYTE in EXECUTION_HISTORY_D520_THROUGH_D536_20261015.md
SHA9aa084e924696efd2ef84879837707b6626a9bdc43282d2f1127d19c002cba05.
Earlier EXECUTION_HISTORY_THROUGH_D519.md SHA
e0ed5415b779effec429b48ea716abde2e3e9b1a379d5abbc11073cdca2f3a99.
Historical running/prepared notes in archives are superseded here.

## Immediate state / next action

D538 targetedstorageDONE; D539 offlinequeueauditDONE. No liveGPU/remotejob.
Read D538_TARGETED_CACHE_RECLAIM.md and D539_SERVERLESSLLM_QUEUE_REVIEW.md.
D538 removedONLYclosedD532duplicateobjects1320files; retainedoriginalpool SHAchecked.
Reclaimed3305652224B,diskafter162489970688B; protected147unchanged.
Auxpeak3222093824B,high3159,max/OOM/kill/swap0; no concurrentperformance.
Actualemptyclosed/pathgone. Rawstorage_maintenance/d538_20261015.
D539 reusedD537/D414nativejournals, no inference/configchange.
16/16D537workerlogs showcompiledmodelgraphcachehits, notallgraphsrecompiled.
Earliest4ready180.412-180.570safterbusiness;routercreated->ready130.962-134.077s.
Dynamo25.91-27.58s,engineprofile/KV/warmup42.30-44.54s (nestednotadditive).
Later12created->ready66.909-75.004s,init20.87-22.80s.
Native4autoscaling/stopfunctionsASTequalofficialcommit9f50241;
routerfileSHAequalsD414. Queue96.794%ofmeanTTFT, notproofuniqueCPU/JITcause.
No evidencednewadaptationdefect;retainD537nonfeasiblepoint,notbaselineparametersweep.
D539auxpeak513466368B,high/max/OOM/swap0,actualemptyclosed/pathgone.
D538/D539curationDONE,twoarchive4938/17311B; curationauxclosed/pathgone,
peak27033600B,events0/swap0. Publicationpreparedexplicitpaths,no productionchange.
D538/D539publicationCOMPLETE:f58f04c764adfece86f5983cee89cc6290cf11ad
PUSHED/remoteequal,16explicitfiles;allchecksPASS. Firstattemptstoppeddiffcheck
atnewdocEOFblankline;correctedonlywhitespacebeforecommit,noexperimentrerun.
Failedpublicationauxclosed26849280B;finishingauxclosed26890240B;events0/swap0.
RAW D539publication_receipt.json/publication_finish_closure.json retainidentity.
No servingcodechangeorBASEcommitthisturn. Postpushstatusdeltafornextcheckpoint.
NextreuseS-LoRAD435nativequalification for7BW1;
continueLoquetier/HydraServeW1. Do notrerunD537/D533/D536orW0.
PrimeLoRAW1still93.725%<95; noG1/G2completion. Allfullscopebelowretained.

D540 S-LoRA W1 OFFLINEQUALIFICATIONPASS, PRELAUNCHPREPARED, NOTLAUNCHED.
RAW originalresults/ieee_tc/slora_qualification/d540_20261015.
inspect_reuse.py/reuse_inspection.json preserved:oldD430viewwholehelperSHA
staleafterLoquetier/HydraServe/W1support,notnativepolicydefect.
OnlytwohelperSHAsdiffer;16S-LoRAmeasurement/sourceadapter/runtimefunctions
ASTequalD435BASEcae13c57. Original/measuredsourceSHAsidenticaltonewsource_view.
Existingprepare_slora_measurement_view reused; native_7b.json differsfrom
frozenBASEconfigonlysource_view/pathmanifestSHA. Noengineconfigtuning/newweights.
http_7b_4000.json differsD435onlyexplicitW1view/profile/experimentrole.
Inputview103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3,
4000requests/132requestedIDs/458224targets;staticfull500preparationretained.
51CPUtestsPASS4.053s inoldqualifiedwrapperenv;bothreuse/qualificationauxclosed.
NoGPUsacquired,no remoteunitsstarted,noformalperformancecompleted.
Launcher/start/stop/evidence/backup helpers adapted from D435/D537 existing recipes.
Read D540_SLORA_W1.md. NEXT explicit prelaunch backup and close its owned scope,
then bounded start.py seals Triton/input metadata, checks fresh resource/remote gates,
and launches ONE7BW1run. Do notrepeatqualification. No baseline algorithm changes.
Usecurrentanalyzer slora_resident_rr_v1/full4000_w1 (alreadyimplemented).
KeeporiginalS-LoRAnativeenv,DP4/TP1/RR,sequentialstartup,packedHOST,nativepaging;
notice+60business,full500remoteafter-notice,coldprivatecache,new/tmp/tc-d540-7bs.
Freshlastdisk162478022656B,MemAvailable110046964kB,GPUnone;notlaunchguarantee.
Allsourceweights/tracesunchanged;51testsnotanewperformanceexperiment.

D537 ServerlessLLM W1 TERMINAL SUCCESS; no live model/remote job remains.
Read D537_SERVERLESSLLM_W1_RESULT.md. RAW originalresults/ieee_tc/serverless_qualification/d537_20261015.
Single4000/nativebinding/458224tokens/0fail. Independent numerical proof UNKNOWN,
n_correctnull. Joint1771=44.275%,TTFT1774,TPOT3994; NOT95%feasible.
U8845.260943896137,R2.2113152359740345/bindingvalidrequest;
P95TTFT109994.30134566501ms,meanTTFT20531.567903198593;
meanTPOT32.25045870216317,P9539.897619416782014,meanE2E24157.993282980577.
Meanrouterqueue19873.30695208849ms=96.793908%meanTTFT;
serviceTTFT462.9181316596223ms. phasejoint0,175,327,87,123,213,362,484.
16actualreadyreceipts, native4GPUstorepool throughout, nofakephysicalscalein.
132fetch/131480060wireB/packing0;bilateralSHAjoinpass; alltimingrecompute0.
GPUwindows pre72.62134733144194,arrival8747.968517303467,
drain2.7068789647892118,cleanup21.96420029643923.
Servicepeak58476834816B,high/max/OOM/kill/swap0; allGPUactuallyreleased.
tmuxfinished, launchservice/watchdog/replayRC0,pathgone,truephysicalrelease.
Ownedauxrecursiveempty/closed/pathgonepeak464056320B/events0/swap0.
D537ownloaderrestored;3remoteunitsinactive/MainPID0/portsabsent.
Analysisclosedpeak609325056B;crossinputclosedpeak318218240B;events0/swap0.
Crosscontract all4000 D537-to-D536 inputs match; sealedD533-to-D536witness
givestransitive3systeminputjoin. Nativeversions/slotrank/cachehistoriesdifferent;
inputidentityNOTinitial-stateidentity or independentnumericproof.
PrimeLoRAvsServerlessLLM U7.912817%lower,P9595.933124%lower,joint+49.45pp;
allthreeW1pointsbelow95,noG1/G2closure. WholeW1referenceNOTformallyfrozen.
D537initialcompile25.91-27.58s/init44.25-44.54s vsD414earlycompile10.77-13.10/
init21.39-23.78s. LaterD537init~22s. Rayaggregatedlogsnotindependentsamples;
notproofallqueuefromcompile, nativeflaw, oradaptationdefect.
OriginalnextactionsnowDONE:backup3e759d7,storageD538,queueauditD539above.
No blindrepeat/configsweep/oldqualification/D533/D536/W0rerun.

Curatedmain_comparison/d537_serverlessllm_w1/result1 DONE:
audit.json,cross_contract.json,startup_diagnostic.json,manifest.json,evidence.tar.gz96390B.
Resultscopeclosed,pathgone,peak28344320B,events0/swap0. Publication COMPLETE:
3e759d75b6e69e182d646d87aca5090f5935024f PUSHED/remoteequal,10explicitfiles.
Archive/SHA/protected147/userdirt/secrets/diff checksPASS;114tests reused.
Resultpublicationscopeclosed/recursiveempty/pathgone,peak31260672B,events0/swap0.
RAW result_publication_receipt.json/result_publication_closure.json retainexactidentity.
No secondbackup/performance/tests needed. This postpushstatusdelta stays fornextcheckpoint.
PrelaunchMAIN3062716be26faf17d607928e8b26b45b5186452a/BASEbde2154,
bothpushed. D414native0102/store/RRconfig retained; onlyW1view/newpaths.
114tests reused112nativeenv+2analysisenv; alloriginalerrors retained, noinstall.
No servingcode/configchanged thisturn. Exactownoverlay/backups/restorationretained.
Actualnotice1535948.902932362,businesst01536008.902932362; noearlydelaysremoved.
D537sourceview4filesmatchD414. Actualservice68a54559dfa545b6bfb8d1b39a50cda7,
Invocation7e3fd03a02cd4ca89d0d63e668ff5666 nowgone.
Status+resultpublication usesexplicitpaths only, protected147 anduserdirtpreserved.
Diskafterpublication159185506304B (~148.3GiB), below150GiBnext-heavy-start,
above100GiBongoingstop. No cleanupduringrun. Beforeanotherheavyjob:
specificclosedrebuildablecacheaudit/reclaim, no broadD483repeat/rawevidencedeletion.
D532andD536privatecaches~3.31GB each retained; do NOTremove byname.
CurrentD537materializedobjectsneededforanalyzersourcevalidation/backup.

D533 TERMINAL SUCCESS and sealed; read D533_FULL_W1_RESULT.md.

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
Post-D537 MemAvailable109910204kB;disk159230144512B. No liveGPUjob.
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
data192.168.4.174ports18080/18081 nowD537STOPPED. Owned3bInvocation
d87d39718a284cd08f2a531eb10410ea;7b23282ad254d44e63989763082fc3fdf1;
monitorprimelora-artifact-monitor-d537-w1.service=a1769aa75a9c460a81945d8bebe1bdb1.
BothhealthPASS prepublished_gzip_v1/artifact_timing_v2, actual1000/full.
7bclockremote-process-monotonic:06a8d3b61e5847a786b5bd64fc72daf3.
Remoteout/home/lab14/primelora_remote/tc/d537_20261015. Privatetoken
/home/qhq/.config/primelora-tc-d75/artifact.token NEVERprint/stage.
D78/D80readonlygzip1581215261B reuse,norepublish/newpool/requestpacking.
Latestactual1Gbps/full;old100Mbpsdistinct,causeunknown.

## Full remaining scope — ACTIVE

D537 newrun7b_repaired/sourceview/http/start/restore/close helpers exist.
Sourceview4measuredfileSHAs equalD414, native configASTunchanged.
W1viewSHA103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3;
4000/132IDs/458224tokens, allsourceassetsreused. Existinganalyzerfull4000_w1.
Loaderoverlay7b_repaired/install.json RESTORED usingownbackups afterrelease.
Do not rerunrestore or executeD414oldpaths/receipts.

7BW1/W2/G2/commonrefs/allpublishedsystems;PrimeLoRAevidence-drivenoptimization;
3Boptimization/allbaselines;dLoRA3BB6;A1–A5confirmedstate/elasticity/handoff/
hierarchy/admission/firstservice;S1–S13tier/mismatch/contention/load/GPU/network/
workload/KV/control/backend/EOS/design;offlinecost/CE/Pareto;B1/B2/C1/C2/C3/
formula/novelty/reviewdocs;lateracademicfigures/V2delivery. No goalcompletion.
