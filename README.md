---
title: TrUcost
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app-V4.py
python_version: "3.12"
short_description: Education finance decisions, made visible.
pinned: false
---

# TrUcost

A self-contained Streamlit decision-support app for education finance: a student
loan calculator, an overseas study cost comparison, saved scenarios with JSON
import/export, and an optional Gemini-backed advisor.

## Run locally

```bash
.venv/bin/streamlit run app-V4.py
```

## Configuration

The advisor needs a Gemini API key. The app resolves it from, in priority order:
a key typed into the running app, Streamlit secrets, the environment, then the
`GEMINI_API_KEY` constant at the top of `app-V4.py`. Accepted names are
`GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `TRUCOST_GEMINI_API_KEY`.

Keep the constant in `app-V4.py` empty. Supply the key out-of-band instead:

- **Local:** `.streamlit/secrets.toml` (gitignored, already set up)
- **Streamlit Community Cloud:** app settings, Secrets panel
- **Hugging Face Spaces:** Settings, Variables and secrets, as a *secret*

Both calculators work fully with no key configured.

## Notes

Saved scenarios live in the Streamlit session only. They do not survive a
restart, redeploy, or idle timeout, so use the per-scenario Export button to
keep anything worth keeping.

Estimates only. Check lender terms, tax rules, visa constraints, and your own
circumstances before acting.
