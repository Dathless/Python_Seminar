# Python Seminar — Django Project

Simple Django project scaffold for a `diabetes` app (SQLite by default).

## Requirements

- Python 3.11+ (recommended)
- pip

## Quick start

### 1) Create & activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirement.txt
```

### 3) Run migrations

```bash
python manage.py migrate
```

### 4) Create an admin user (optional)

```bash
python manage.py createsuperuser
```

### 5) Start the development server

```bash
python manage.py runserver
```

Then open:

- App: `http://127.0.0.1:8000/`

## Project structure

- `config/` — Django project settings and root URLs
- `diabetes/` — Django app
- `db.sqlite3` — Local SQLite database (development)

## Known issue (URLs)

`config/urls.py` includes `path('diabetes/', include('diabetes.urls'))`, but `diabetes/urls.py` is not present in this repo snapshot.

If you get an error like “No module named 'diabetes.urls'”, add `diabetes/urls.py` (and corresponding views) or remove that include while developing.

## Notes for GitHub / production

- `config/settings.py` currently contains `SECRET_KEY` and `DEBUG = True`. For real deployments, move secrets to environment variables and set `DEBUG = False`.
- `ALLOWED_HOSTS` is empty; set it for deployment.
