# Continuous training and policy delivery

Status: IN PROGRESS. Remote credentials and SSH access are pending.

| Requirement | Status | Implementation / validation |
|---|---|---|
| Public object storage, private publication credentials | IN PROGRESS | training/publication.py; dedicated poker-policies bucket |
| Hourly exports without checkpoint reload | DONE | training/vps.py; existing VPS regression tests |
| Independent retryable uploader and bounded remote retention | IN PROGRESS | publication service/timer; failure and ordering tests |
| UI and worker poll versions, validate hashes, release old sessions | TODO | shared policy catalog and model bank |
| Automated tested code deployment, checkpoint-compatible resume | TODO | GitHub Actions and VPS release script |
| Local end-to-end export/upload/download/inference | TODO | bounded real model and HTTP storage contract tests |
| Real Supabase provisioning/upload/download | BLOCKED | Awaiting server-only credential file |
| Real VPS installation and memory validation | BLOCKED | SSH blocked; do not probe until access is restored |

One writer per track. HU and 3-max have separate pointers, so independent hosts do
not overwrite each other's catalog entries. Public models are experimental learned
policies; publication validates compatibility and inference, not poker strength.
