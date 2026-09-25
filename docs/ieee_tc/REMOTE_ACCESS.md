# Remote artifact node: execution record

## Verified 2026-09-25

- User-authorized dedicated Ed25519 key deployed, restricted against SSH
  forwarding/PTY features. No password/private key is stored in this repository.
- Host identity matched the pre-existing trusted key for
  `[192.168.4.174]:8122`, fingerprint
  `SHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40`.
- Management alias: `primelora-artifact-174`; account `lab14`, port `8122`.
- Local private key: `~/.ssh/primelora_artifact_174_ed25519_20260925`, mode 0600.
- `ssh primelora-artifact-174 'id -un'` succeeds with BatchMode/strict host
  checking and no password interaction.
- The management IP `10.199.227.174` did not answer the short SSH probe.
  The already-trusted experiment LAN address `192.168.4.174` works and refers
  to the same approved node. HTTP data remains direct on the experiment LAN.
- User systemd manager reports running; no artifact listener found on 18080/18081.
- Original server exists at the approved path; model pools are reused in place.

## Pending gate (do not silently override)

Remote available disk measured 149,137,707,008 bytes (138.9 GiB), below the
approved 150 GiB floor. No disposable artifact temp directories were found.
Existing full-pool archives total under 2 GiB; deleting them cannot resolve
the shortfall. Other projects and unique 13B assets were not touched.

A user question is pending about retaining the common floor versus approving
a separate measured-peak rule for the lightweight artifact node. Meanwhile,
local audits continue. No remote artifact service was started, and full
start/stop/restart/download qualification is NOT yet complete.
