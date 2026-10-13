# IEEE TC status — D495 native 7B delivery and real app initialization passed

Updated2026-10-13. Current checkpoint, NOT replacement plan. Entire preceding
status byte-identically archived EXECUTION_HISTORY_THROUGH_D491.md; cmp passed,
SHA017a48a3675c0aa9e2f1baff09552539a0ebe64123816a2ad883f25a7f5063d1.
Older D477/D461/D448/D433/D402 histories remain. Their NEXT/RUNNING notes are
historical, not current instructions. No approved task/outcome/failure removed.

## Authority and main line

Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md before work,
after compaction and before experiments; SHA
f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5.
Also BASELINE_COMPLETION_ORDER_20261012.md, BASELINE_EXECUTION_RULES_20261011.md,
STORAGE_PRIORITY_20261013.md. D483 maintenance COMPLETE47,340,646,400 allocatedB
reclaimed; no repeated broad storage sweep by default. D486 metric recheck DONE.

LATEST ORDER: faithful HydraServe actualresult -> all availablebaseline comparison
-> PrimeLoRA optimization. No new PrimeLoRA candidate now. Only PrimeLoRA
performance optimization; baseline compatibility/correctness/measurement/safety
only, no performance tuning/configsearch. Existing k3s belongs to another project;
do not access/change cluster/credentials/permissions/resources. Isolated rootless
Podman adaptation authorized; no barevLLM substitution, fakeReady or fake nodes.

## D494–D495 current checkpoint (NOT a performance result)

BASE c334727892b7b7fa2c63c4fd10cf3f124f9695a1 pushed/independentlyremoteequal.
MAIN preceding9446e814ebdde563878cc9b156028756663f6441; currentstatusbackup follows.
Read BASEdocs/ieee_tc/D495_HYDRASERVE_NATIVE_DELIVERY.md. No liveheavyjob.
Actualnative CPUprovider/two-existing-HFshards/packing/two-threadTCP/private13GiB
SHM/C++GPUfastload/originalapp+LoRAmanager/KVinit allconnected. D495 /init200,
/health200,/v1/models200. OriginalResourceManager actualGPUselection/release used;
GPU-visibleUUID verified beforepermit. Onephysicalhost, notfour inventednodes.
Readonly metadata/symlinks only; no tensorcopy/combinedcheckpoint/newdataset.

D494 originalfailure retained: nativebatchdefault65536 violates ownLoRAkernel
ceiling65528, so/init500. D495 explicitlyuses source-AST-derived65528 onthree
LoRA-enabledEngineArgs sites; notparametersearch/performanceoptimization.
Officialconfig re-fetchedonline, SHA5a17e5ca3741314175daa825024172be0decb56dd411d6e40ff146cfe15d8332.
ProviderwithoutLoRAdefaultunchanged; originalpacking/wire/storagepolicies retained.

Bothrunscommonhigh72/max80GiB/swap2,CPU4-23/28-47; wholeprovider/storage/worker
incommonparent. D494peak42040569856B; D495peak42318667776B; allhigh/max/OOM/
OOMkill/swap0. Workers/provider/storage/pausegone; actualGPUclear,svcgroupgone,
exactInvocation/inode emptyauxclosed. D495svcInvocation303886b8151e43debae95deca14549cf,
inode23993694; auxInvocationeacea1d57eaa4b42a66c8c7dfb4d335a,inode23993652.
45CPUregressionsPASS; protected147/userdirty/nativeclean/secrets/diff/PUSHallPASS.
186explicitfailure/success/source records roundtripSHAchecked:
BASEpaper_results/ieee_tc/hydraserve/d495/native_delivery_evidence.tar.gz SHA
7cc57b5e05fc09e9fb34f76abe787955c549d1f67ae9578ba3e725fc4e62fbf4;
summarySHA526a43234a21ad053d196eeea79d597e3c3f1439fe1754b42bd628cc4bbc3a98.
Raw originalresults/ieee_tc/hydraserve_qualification/d494_20261013,d495_20261013.
Latestdiskfree201970425856B(~188.1GiB); available110223124KiB; freshgates required.

NEXT: nativeRequest/ChatEngine input+LoRA+fixed-output+native-token/timingcontract,
not another elementary import/container/SHM test. Officialsrc/engine.py chat uses
temperature1/max_tokens1024, Request in src/scheduler.py lacksadapter/target.
Native vllm/entrypoints/openai/serving_completion.py alreadyhasLoRARequest and
token_ids; keep actualHydraServe scheduler, don't replacewithbarevLLM. Registration
llama2_7b hasnochattemplate; choose necessarycanonical-input adaptation explicitly.
Also observed nativeapp startup asyncio.run closesoldloop; end_load_model_shm
laterreports Event loop is closed. Storagefree_buffer observedonlyaroundworker
deletion; notproof of promptsteady-stateSHMrelease. Trace control-socketownership
before multireplica/consolidation; protocol-preserving correctnessrepair ifneeded.
Nativeactualgeneration/per-requestLoRA/pipeline/KVmigration/concurrentconsolidation
remainOPEN. Then100->500-ID->1000->4000 HydraServe -> availablebaselinecomparison
-> PrimeLoRAoptimization. No newperformancepoint/ranking/claimfrominitialization.

## D492–D493 completed (NOT inference/performance)

BASE929b18ad1f84b8d3433de03126430794c8ae707e pushed/independentlyremoteequal.
MAIN predecessora108123ff7a5bf6b9976683952e02c70e79358d3 remoteverified;
status/history backup follows, exactreceipt D493raw. No live heavyjob.
Raw originalexperiment results/ieee_tc/hydraserve_qualification/d492_20261013
and d493_20261013. Read BASEdocs/ieee_tc/D493_HYDRASERVE_CONCRETE_LIFECYCLE.md.

NEW HydraServe_project/runtime/isolated_platform.py actually implements D480
LifecyclePlatform, reusing D477/D481/D491 containment/IPC/address recipe.
stdlib-only container_entry.py awaits actualPID/birth/affinity/limits/privateIPC
checks BEFORE nativecommand; no torch import before original app. One physical
host inventory; explicit native-image mapping and immutable localimage, readonly
mounts and selectedUUID. Original ResourceManager still chooses GPU, creates,
awaits /init and releases. Addresses are NOT Ready; deletion requires actual
exit/recursiveempty childgroup. Hostnetwork is NOT kernel network isolation.

D493 ran original local_server with D481 address-only overlay, actual1GiB
privateSHM allocation/touch and originalhealth/free PASS. Original ResourceManager
created TWO realcontainers,gpu_id0/1; both read privateobject and nativehealth.
Synthetic NO-MODEL /init endpoints delayedacks; controller actuallywaited then
restored resourcebookkeeping only afteractualremoval. NOT modelReady/inference/
fastload/GPUarithmetic. CDI rewrites NVIDIA_VISIBLE_DEVICES to void while
injecting selectedGPUdevicefiles; don't infer noGPU from thatenv. NoCUDA/model.

Allworkers/storage/pausegone; scope1Invocation4fc23cef1c1f485f8c865213ec3e8cec,
inode23976812 actualrecursiveemptyclosed. Parenthigh3/max4GiB/swap0,
CPU2,3,26,27; childmax2GiB/swap0. Peak1450344448B,allhigh/max/OOM/OOMkill/swap0.
Free201994956800B(~188.12GiB); freshgatesneeded. NOT formalserviceenvelope.
D492 failed BEFOREcontainers because baseline scripts shadowed mainpreflight.
D493 imports preflight FIRST then addsBASEpath. Originalcode/error retained;
no claimedoriginalexitcode. ScopeInvocationf21ceb89e2bb4681a350dcdc49878e50
actuallyemptyclosed. NotGPUfailure/OOM/baselineperformancefailure.

Afteractualrun: entryJSON atomicpublication and boundedCLI cancellation awaiting
completion beforecleanup added. Originalexecutedsources retained *_executed.py;
don't claim laterrevisions were exercised inD493containers. Updatedcode38CPUtests
PASS includingactualboundedsubprocesscancel, priorpolicy/source andD479/D491tests.
Testscope2closedpeak27721728B/events0.95explicit source/success/failure/resource
records archived/roundtripSHAverified; protected147/userdirty/nativeclean/secrets/
diff/explicitstagePASS. BASEpaper_results/ieee_tc/hydraserve/d493:
archiveSHA0f025ce5fbfa4fe3c67d58e34d2c3020af8388ed2e72fa58b83db7117ae05ff5,
summarySHAa8ea86c80a3407d7007f014f67186cea027cadadbe3a114fea15423aa57bfb0d.
Curation3actualemptyclosedpeak12320768B/events0; backup4closurefollows.
Officialresource_manager.py refetchedonline SHA equals pinnedinput. Webtool429
notusedasevidence; boundedrawGitHubfetch succeeded. No nativecheckout changes.

## D494 precursor detail (D495 current next action above)

D494 (2026-10-13) NEW actual CPU provider loaded existing TWO 7B HF shards,
native packing/partition/two-thread wire -> private13GiB SHM -> C++ GPU fastload
completed. GPU UUID gate actually passed. Original app /init returned500 before
engine construction: native batch default65536 exceeds its own LoRA kernel cap
65528. This is a concrete compatibility failure, NOT a performance result.
Raw d494_20261013 retains executed source overlays, logs,failedresult and closure.
Common72/80GiB,swap2,CPUs4-23/28-47; observedpeak42040569856B,events/swap0;
GPUreleased/servicegone/exact-owned emptyauxclosed. Existingk3s untouched.
NEW readonly metadata links/shard iterator/import overlay/provider lifecycle
passed15CPUtests. No wholecheckpointcopy/download/partition rewrite. Nativeapp
LoRAenable/extra_vocab0 necessary for existingweights; native LoRA/batch/cache
defaults otherwise retained. NEXT revision only clips batchtokenlimit to exact
source-pinned native LoRA ceiling65528 (not performance tuning); D495initnowpassed.
Upper requestLoRA/nativeoutputcontract stillOPEN. No PrimeLoRA candidate now.

Use concreteplatform with D491lora_worker_binding overlay. Preserve modified
vLLM0.4.2/nativepipeline/startupoverlap/consolidation. Need actualprovider->private
SHM->C++fastload->app/init (D495passed) AND upperleveladapter/fixed-output/native-tokencontract.
Then100->500-ID->1000->4000 HydraServeactualresult -> fullavailablebaseline
comparison -> PrimeLoRAoptimization. Do NOT repeat install/build/elementaryCUDA/
container or sharedbinding tests instead of advancing this integration.

Newsourcefacts (readoriginals beforeimplementation):
- remote_server defaultsmodel_set3 including13B; restrict explicitexisting7B/3B
  registration. NativeModel CPU-packed byte strings+retainedCPUmodel duplicate
  memory; accountincommon80GiB, preserve nativepartition/serialization/two threads.
- provider's offline safetensors_weights_iterator reads ONLY MODEL_PATH; existing
  7B hasTWOHFshards. Necessaryreadonlymulti-sharditerator adaptation; no combined
  fullcopy. NEVERdownloader.partition onuniqueweights (pp1rewrites/deletes).
  Onlysmallconfig/tokenizerlinks/partitionmanifest/duplicateJSON metadata views.
- LOCAL_SERVER=1 FAST_COLDSTART=1 ENABLE_PARA=0 ISoriginalSHMfastpath.
  App starts torch.ops.start_load_shm BEFOREvLLMimports. Wrapper mustnot addheavy
  torchimport ahead. ThreeEngineArgs siteslackLoRAenable; upper Request/ChatEngine
  lacksadapter/generation/nativeclockcontract; low-levelsequence/broadcast hasLoRA.
- local_server touchesSHM_SIZEGiB andclearsprivatepsm_*; neverhostshm. This1GiB
  test isNOTfull7Bcapacity. Actualprovider+SHM+workers need80GiB audit+watchdog.
- native slow_exprdefault0/USE_CACHEdefault0 preserved. Scheduler acts perPHYSICAL
  NODE, so onehost4GPUs naturallyselectspp1; no inventednodes or forcedconsolidation.
  Controlled pipeline/KVmigration/concurrentconsolidation correctness stillOPEN.

## Qualified components and reusable environment (don't redo)

BASE/vendor_new_baselines/HydraServe_main_20260520 cleanHEAD
8ae605de354ccfa2e9095514cdb4a9e9c56aa56b. D480/D481/D491 sourcepinned overlays.
BASE/installs/hydraserve_native_d484_20261013: Python3.10,Torch2.3.0+cu121,
xformers0.0.26.post1,Transformers4.43.1,flash_attn2.5.8/Torch2.3/cp310/ABIfalse,
Triton2.3.0,Ray2.9.3,nativevllm0.4.2 originalPython+ownD485kernels.
ReadonlyPythonbase anaconda3/envs/hydraserve_py_20260520; readonlyCUDA12.4.131
andnvidia namespace from slinfer_official_20260612. Neverpip-upgrade sharednamespace.
D486private_home/.config/vllm/nccl/cu12/libnccl.so.2.18.1 SHA
005503b62b9fe466b6b2f60ba2ad1daa4571196ffcb82a3d77a3c775bfcd5ac8;
setVLLM_NCCL_SO_PATH. D485build_fastload/lib/libst_pybinding.so SHA
f11b31a82f654d7c8950d85e05461d31b5cc45469720deb1e784d47022571e44,
mount/vllm-workspace/libst_pybinding.so. ImportPASS, actualSHMload/appinitD495PASS;
end-to-endgenerative correctnessstillOPEN.

D487Punica/SiLU/FlashAttention GPU arithmeticPASS; D488nonzeroLoRA128positionsPASS.
D491 actualnativeDefaultModelLoader/Llama/ModelRunner2loadboundaries/oneLRUmanager
pp4rank3->pp2rank1 ONONEGPU,32packedmodules global16–31/twoalreadyactiveLoRA;
maxrelativeL2.0002201656752731651,fixed.005tolerance;8decoderobjects/head/slots
shared. CompleteoriginalCPUinit/model.to/postprocess, onlyHFiteratorcontrolled.
NOTattention/KV/pipeline/KVmigration/concurrentconsolidation/fastload/e2etokens.
ReadD491doc; D489circularimport/D490incompleteTESTfactoryfailures retained.

Ubuntu22.04imagee8df6c74f650ad385efe2e20103eca9b0ad50690fb3c63d49fa00762412bbf55,
D477container_proxy_attempt/container_storage vfs; runtime
/run/user/1001/primelora-hydra-d477-proxy/runroot; Podman3.4.4/NVIDIAruntime1.19.
Use/usr/bin/python3 hostpidfdcleanup. Pauseinitiallyownscope: verify/adoptBEFORE
workers, notclaimborninparent. CPUcontrollerundelegated; taskset+actualaffinity.
ReuserecipeD493/executed_probe.py plusBASE D480/D481/D493docs.
Originalbackbone models/meta-llama--Llama-2-7b-hf readonly. financefullSHA
971901c3140f83f57cbb19fcc86d33b4d010f6d8fc15fbec621f0347bc31a258;
medical d2b1dcfe42193917d0b2e0f728e735f24ad119dd2c77cb9b20c983c7f688fa5c.
EqualheaderSHA isNOTequalfullweights.500logical=498zero/2nonzero,4weightSHA/
6treeclasses limitationretained. No newmodels/traces/pools/13B.

## Frozen evaluation and sealed7BW0 (do not repeat)

Beforecomparison/configselection read METRIC_PROTOCOL_FROZEN_V1.md and
METRIC_PROTOCOL_SINGLE_RUN_V2.md FULL; respectiveSHAs
5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22,
193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d.
G1 all4000contracts+commonjointTTFT/TPOT>=95%,thenphysicalGPU-s. G2same+
U<=.75Uref,W1/W2tail/observedSLOdifference; CEsupplementary. Unknownnumerical
n_correctstaysunknown. Oneperformance/frozenkey; noCI/repeatmeans/winnerfilter.
TPOTrelative regressionalone notfailure; retain development/reuseidentity.
Sealed(joint,GPU-s/boundrequest,P95TTFTms):
PrimeLoRA D476 .97425/2.694612200556614/2315.363336;
ServerlessLLM D414 .98125/4.022440686/1052.246361;
vLLM D423 .99475/4.028844291/1200.506860;
S-LoRA D435 .94425/3.789269468/17699.067366 nonfeasible;
Loquetier D445 .0435/4.249595809/555110.032229 nonfeasible;
ElasticLocality D453 .95775/2.473178968/2420.607829.
HydraServe missing; PrimeLoRA stilllosesElasticLocalityresource. No newcomparison.
D3947BSLO input<=616/>616:TTFT2995.348713684507/4633.345350589479ms,
TPOT59.74409694902162/85.24698166015581ms.W0Uref16115.377162149176,
B12086.532871611882,Qref1200.5068596918136.W1/W2refsOPEN; qualifyactualW1
transform, not relabeldefaultW0. W0 isNOTW1/W2/G2closure. Historiesretaindetails.

## Safety, protection and publishing

Oneheavyjob; no experiment withbuild/install/hashscan/compression/remotechanges.
Diskstart>=max150GiB,100GiB+1.5growth; runtime100GiB. Servicehigh72/max80GiB,
swap2,CPU4-23,28-47; auxhigh3/max4GiB,swap0,CPU2,3,26,27. Emptyservicestart
MemAvailable>=102GiB; otherwise remaininggrowthformula. Actualworker restriction
BEFOREmodel/external16GiBhost/100GiBdiskguard. No blindOOMretry/globalGPUreset/
reboot/driver/unknowncleanup. Close exactInvocation/inode+actualrecursiveempty,
never virtualfilesize or systemdactivelabel. No rawlargefiles/credentials/forcepush.

Protected147seal originalresults/ieee_tc/slora_qualification/d435_20261011/protected_before.json
SHA79df0c64c0d88236bc20b439d735f8dcab113f06582421151742aa63a5a2be54.
Neveredit/stageMAINconfigs/generated/lora_manifest_1000.json SHA
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df.
BASEscripts/replay_openai_trace.py SHA
04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22;
scripts/run_serverlessllm_relayserve_continuation.sh SHA
337dceec7a528b7816e5a55e81d3c26f3feb1499976a4d049faa69809acec499.
Explicitstage/source/checksum/tests/secrets/diff; MAINfaaslora_origin/
retry14_continuous_queue_v2 andBASEorigin/main; verifiedremotebackupcheckpoints.

RemoteSSHprimelora-artifact-174 lab14@10.199.227.174:8122 strictknownhost;
data192.168.4.174ports18080/18081inactiveD476.TokenoutsideGit
/home/qhq/.config/primelora-tc-d75/artifact.token NEVERprint. D78/D80readonlygzip
1581215261B reuse,no republish/extractedpool/requestpacking. Latest1000Mb/s/full;
old100Mbpsdistinctstate,causeunknown. Realnecessaryread/transfer/contention/
materializationcount; no oldspan subtraction. No remotechangesduringexperiments.

## Remaining fullscope — goalACTIVE

HydraServeactualresult/fullavailable7Bcomparison/PrimeLoRAoptimization;
7BW1/W2/G2/commonrefs;3Boptimization/baselines;dLoRA3BB6;A1–A5confirmedstate/
elasticity/handoff/hierarchy/admission/firstservice;S1–S13tier/mismatch/contention/
load/GPU/network/workload/KV/control/backend/EOS/designsensitivity;offlineCE/cost/
Pareto;B1/B2/C1/C2/C3/formula/novelty/two-reviewdocs;lateracademicfigures/V2delivery.
NineIEEEequations unchanged. No13B/newSGLang/paperedits/newweights/trace/pools/
oldresultoverwrite. Nofiguresuntilneargoal. Chinese paper-evidence updates, full
names PrimeLoRA/ServerlessLLM. Setup/lifecycle checks areNOTperformanceexperiments.
