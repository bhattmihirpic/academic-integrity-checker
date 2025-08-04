#!/usr/bin/env bash
# One-click start for Academic Integrity Tool

# (Optional) Activate your Python virtual environment if present
if [ -f "integrity_env/bin/activate" ]; then
  source integrity_env/bin/activate
fi

# Export environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export INTEGRITY_ENV=production   # or set to "development" as needed

echo "🗄️ Ensuring database tables exist..."
python - <<'PYCODE'
from app import db, app
with app.app_context():
    db.create_all()
    print("✅ Database initialized.")
PYCODE

echo "🚀 Starting Academic Integrity Tool..."
flask run --host=0.0.0.0 --port=5000
