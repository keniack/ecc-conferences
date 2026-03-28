# 🌍 Edge and Cloud Conferences CFP Tracker

This repository tracks **major Edge and Cloud computing conferences**, including their **deadlines, rankings, and details**.

📅 Stay up to date with the latest submission deadlines and notifications.  
🔍 Easily browse upcoming and past conferences.  
🚀 Hosted at:  
➡️ **[https://keniack.github.io/ecc-conferences/](https://keniack.github.io/ecc-conferences/)**

---

## 📌 Features
✔️ **Conference List** (Upcoming & Past)  
✔️ **Sorted by Submission Deadline**  
✔️ **Days to Submission Countdown**  
✔️ **Ranks & Conference Details**  

---

## 🏗 How to Contribute

To contribute, update the **`_data/conferences.yaml`** file with new or modified conference details.  
After editing, commit and push your changes, then submit a **pull request**.

## 💻 Run Locally

This site is a Jekyll site served by GitHub Pages.

### Prerequisites

- Ruby
- Bundler

### Install dependencies

```bash
bundle install
```

Dependencies are installed into `vendor/bundle` because of the repo's Bundler config.

### Start a local server

```bash
bundle exec jekyll serve
```

Then open:

```text
http://127.0.0.1:4000/
```

Jekyll will rebuild when you edit `index.html` or `_data/conferences.yaml`.

### Run a local production-style build

```bash
bundle exec jekyll build
```

The generated site will be written to `_site/`.

### Useful files while testing

- `index.html`: homepage UI
- `_data/conferences.yaml`: conference data source
- `_site/index.html`: generated output after a build

If Jekyll prints a `GitHub Metadata` authentication warning locally, that is usually harmless for this site.

## 🤖 Automated Updates

This repo can now run a scheduled agent that checks CFP pages and proposes updates to
**`_data/conferences.yaml`**.

What it does:

- Fetches each conference CFP page
- Tries to detect stale or broken links
- Checks whether submission deadlines were extended
- Optionally searches for a replacement CFP URL when the current page looks outdated
- Uses an LLM to extract structured dates and location data
- Opens a pull request instead of pushing directly to `main`

The updater is intentionally conservative. It only auto-edits these fields:

- `submission_deadline`
- `notification`
- `conference_start`
- `conference_end`
- `location`
- `website`

### GitHub Setup

Add these repository settings before enabling the workflow:

1. Repository secret: `OPENAI_API_KEY`
2. Optional repository variable: `OPENAI_MODEL`
   Default: `gpt-4.1-mini`
3. Optional repository variable: `OPENAI_BASE_URL`
   Only needed if you want an OpenAI-compatible endpoint instead of the default API URL
4. Workflow permissions: set GitHub Actions to `Read and write permissions`

The workflow file lives at **`.github/workflows/conference-agent.yml`** and runs weekly,
plus on manual dispatch.

### Local Run

```bash
python3 -m pip install -r scripts/requirements-conference-agent.txt
python3 scripts/conference_agent.py \
  --search-fallback \
  --report-file /tmp/conference-agent-report.md
```

Without `OPENAI_API_KEY`, the script still works in check-only mode and reports entries that
need review, but it will not rewrite the YAML file.
