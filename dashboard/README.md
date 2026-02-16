# Claims Dashboard

This folder contains:

- `backend/`: Flask API with local JSON storage and a listener that runs the orchestrator.
- `frontend/`: React + Vite dashboard UI with user, claims list, and practitioner routes.
- `data/`: local storage for `claims_db.json` and uploaded submission folders.

## Backend setup

```bash
cd dashboard/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Frontend setup

```bash
cd dashboard/frontend
npm install
npm run dev
```

### User route
- UI: `/submit`
- API: `POST /api/claims`
- Accepts multipart form with:
  - `reports`: one or more files
  - `comments`: optional text
- Stores reports and creates/updates JSON DB

### Claims list route
- UI: `/claims`
- API: `GET /api/claims`
- Shows all claims and current status (`accept`, `reject`, `uncertain`, or processing states).

### Practitioner route
- UI: `/practitioner`
- API:
  - `GET /api/claims/uncertain`
  - `PATCH /api/claims/<submission_id>/practitioner-review`
- Lists uncertain claims and allows practitioner to add final comment and update status.

## Final claim evaluation status:
  - `reject` if any report evaluates reject
  - `accept` if all reports evaluate accept
  - `uncertain` otherwise
