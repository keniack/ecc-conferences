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
✔️ **Homepage Submission Form** that opens a PR through GitHub Actions  

---

## 🏗 How to Contribute

To contribute, update the **`_data/conferences.yaml`** file with new or modified conference details.  
After editing, commit and push your changes, then submit a **pull request**.

You can also use the **Add a conference** form on the homepage. It opens a structured GitHub issue,
and a workflow converts that issue into a PR against **`_data/conferences.yaml`**.
That form only asks for a conference name or acronym plus a public URL. The workflow then tries to enrich
the rest of the fields automatically and falls back to safe placeholder defaults when it cannot extract them yet.

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

This repo also has a separate issue-to-PR workflow for manual conference submissions from the homepage form.
It attempts to fill missing metadata from the submitted page before opening the PR.

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
   Default: `gemini-2.5-flash`
3. Optional repository variable: `OPENAI_BASE_URL`
   Default: `https://generativelanguage.googleapis.com/v1beta/openai`
4. Workflow permissions: set GitHub Actions to `Read and write permissions`
5. Keep GitHub Issues enabled if you want homepage conference submissions to open PRs automatically

The updater workflow lives at **`.github/workflows/conference-agent.yml`** and runs daily,
plus on manual dispatch.

The homepage submission workflow lives at **`.github/workflows/conference-submission.yml`**.
It listens for structured GitHub issues created from the site and opens a PR with the proposed conference entry.

### Local Run

```bash
python3 -m pip install -r scripts/requirements-conference-agent.txt
python3 scripts/conference_agent.py \
  --search-fallback \
  --report-file /tmp/conference-agent-report.md
```

Without `OPENAI_API_KEY`, the script still works in check-only mode and reports entries that
need review, but it will not rewrite the YAML file.
