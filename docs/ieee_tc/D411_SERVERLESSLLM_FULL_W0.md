# D411 ServerlessLLM complete W0 observation — running

Started2026-10-11 03:12:24 Asia/Shanghai, tmux `tc-d411-7b`.
Do not restart this key. No complete result yet; one heavy job only.
Baseline execution commit `2a5dfd1154ba93c82d9b7076fde0a8521543933b`;
main execution `896af899396983350c4544a9d0277283bc28339d` (pushed/remote verified).
No baseline source code changes for this run. Existing runner already accepts
the full4000 trace; main analyzer now explicitly verifies count/source_count,
W0/rate1 and exact request identity.19CPUtests passed under4GiB/zero-swap.

Source42/development41/nonformal, original4000W0/no remapping;132logicalIDs,
458224targettokens,3963.908903s arrivalwindow. Fixed_length_greedy_v1,
common60snotice/openloop/1800squalification protection and D394 commonSLO.
Same D410 native vLLM0.10.2 FP16/store/RR/repairedpolling/identityqueue,
min1/max4/target4/keep10,maxseqs4/maxLoRAs4/CPULoRAs4/tokenbudget1024/
GPUfraction.72,enforce_eager=false. No parameter sweep or100/1000 repeat.

Raw root `/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d411_20261011`.
Service `primelora-tc-svc-cdae00875efc4791b2fad4ef286ef88f.scope`,
InvocationID426554ae3dcf4dd2999ceeecf4a2e012;aux
primelora-tc-aux-d4110000000000000000000000000001.scope.
Deploymentnotice1166817.112284243,t0=1166877.112284243.
RemoteD78/D80 immutablepublishedcache reused,actual1Gbps/full,
7Bclockremote-process-monotonic:158e79ec85e345c19ea6fb6250184f23.
3B/7B/monitorInvocationIDs3dadc35fa0e8430d8ef0a231e943546c,
f5a284ee1ad643d6b4417e3987500387,43c6e563e84f43689b0b27a570ea1255.
No publication/requestpacking/liveadministration. ColdLoRAcache,retainednative
compilerDISKcache:3858files/104734877B,SHA
5a75a00e9e341e6ee53e5afb2195c4a193a010c6d3446adcd7c4dfc10ca50791.
Allactualstartup/capture/physicalholdcharged;not GPU/HOST/KV warmstate.

Safety unchanged72/80GiBservice+2GiBswap,4GiBaux+zeroswap,CPUdomains,
150GiBdiskstart/100GiBstop. D410 only verifieddownloadobjects reclaimed;
rawlogs/originalpool/remoteimmutablecache retained. Protected147entries
unchanged,sole historicaluserreplaymismatch. Userdirtyfiles NOTstaged.

Afterterminal:confirmGPUrelease,restore exactoverlay fromD411installreceipt,
stop remote ownedInvocationIDs usingD411stop_remote.sh,preserveworker/resolver/
remoteevidence,audit full4000withD394,save/backup before next7Bbaseline.
Preserve all failures/unknowncorrectness flags;reuse D406/D409 bounded evidence.
No same-key rerun to win. PrimeLoRA D371 stayssealed; W1/W2/G2/3B/ablations
and sensitivity remain open. Authoritative fullplan and main EXECUTION_STATUS
must be read before continuation,not replaced by this checkpoint.
