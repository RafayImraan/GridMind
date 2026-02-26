# GridMind Demo Readiness

## 1) Live Demo URL

- Frontend (live): `https://<set-your-frontend-url>`
- Backend API (live): `https://<set-your-backend-url>`
- Health check: `https://<set-your-backend-url>/health`

## 2) Offline Fallback Plan

Dashboard includes built-in mode switch:

- `Live API`: uses backend endpoints.
- `Offline Demo`: renders local fallback assessment payloads and keeps demo flow unblocked.

If live API fails mid-demo, switch to `Offline Demo` and continue walkthrough.

## 3) 90-Second Backup Video Checklist

Record one continuous 90-second clip covering:

1. Open dashboard and show `Live API` mode.
2. Submit high-risk assessment input.
3. Show system scores, cascading panel, geo district layer, and executive summary.
4. Scroll to Model Evidence Board (PR-AUC benchmark + calibration artifacts).
5. Toggle to `Offline Demo` mode and rerun to show resilience.

Output files:

- `docs/assets/demo_backup_90s.mp4`
- `docs/assets/demo_backup_90s.webm` (optional)

## 4) Live Pitch Failover Script (10 seconds)

"If connectivity is unstable, GridMind automatically continues in offline mode with identical workflow and precomputed evidence artifacts, so evaluation can proceed without interruption."
