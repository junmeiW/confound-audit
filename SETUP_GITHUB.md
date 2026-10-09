# Publishing `confound-audit` to GitHub

Two steps: create the repository, then push. Everything below is written for
your environment (no `gh` CLI installed, no SSH key configured).

---

## 0. Before anything: two things to fix

### a) Set your git identity

Your current settings are:

```
user.name  = (unset)
user.email = yourgitemail@example.com     <- placeholder
```

Commits with a placeholder email are **not linked to your GitHub account** and
will not appear on your profile. Set a real identity:

```bash
cd /Users/junmei/Documents/aiwork/esm-project/confound-audit

# this repository only (recommended here — leaves the analysis repo unchanged)
git config user.name  "Junmei Wang"
git config user.email "you@example.com"      # <- put a real address here
```

Use the address you registered with GitHub. To keep it private, use GitHub's
no-reply form instead — find your numeric ID at
<https://api.github.com/users/junmeiW> (`"id"` field), then:

```
<ID>+junmeiW@users.noreply.github.com
```

Because `confound-audit/` may not yet be its own repository, `git config`
inside it currently writes to the **parent** repo. That is harmless, but if you
prefer to scope it strictly, run the script first (it initialises `.git`) and
set the identity afterwards, then re-run.

### b) Keep the package out of the parent repository

`confound-audit/` currently sits **inside** the `esm-project/` git repository
and is untracked but *not* ignored. If you ever run `git add -A` in the parent,
git will add it as an **embedded repository** — a pointer whose contents are
silently missing from any clone. This warning is easy to miss:

```
warning: adding embedded git repository: confound-audit
hint: You've added another git repository inside your current repository.
```

Prevent it by adding one line to the **parent** repo's `.gitignore`
(`/Users/junmei/Documents/aiwork/esm-project/.gitignore`):

```gitignore
confound-audit/
```

Then, if it was already staged in the parent:

```bash
cd /Users/junmei/Documents/aiwork/esm-project
git rm -r --cached confound-audit 2>/dev/null || true
```

---

## 1. Initialise and commit

The script does the work safely: it checks for build junk, refuses to commit
with a placeholder email, and runs the test suite first.

```bash
cd /Users/junmei/Documents/aiwork/esm-project/confound-audit
chmod +x scripts/init_repo.sh

# review without changing anything
./scripts/init_repo.sh --dry-run

# create the commit
./scripts/init_repo.sh
```

Expected: **28 files** committed, ~3,450 lines, tests passing.

Do **not** commit `dist/`, `__pycache__/`, or `.pytest_cache/` — the
`.gitignore` already excludes them and the script verifies this.

---

## 2. Create the repository on GitHub

Pick one of two authentication methods first.

### Option A — HTTPS with a token (simplest, no SSH setup)

1. Create a **Personal Access Token (classic)** at
   <https://github.com/settings/tokens> with scope **`repo`**.
2. Create an **empty** repository at <https://github.com/new>:
   - Name: `confound-audit`
   - Visibility: **Public** (so reviewers and the GPB data-availability
     statement can reach it)
   - **Do not** initialise with README, .gitignore, or licence — you already
     have all three, and initialising creates a conflicting history.
3. Push:

```bash
cd /Users/junmei/Documents/aiwork/esm-project/confound-audit
git remote add origin https://github.com/junmeiW/confound-audit.git
git push -u origin main
```

When prompted for a password, paste the **token**, not your account password.

### Option B — SSH key

No key exists on this machine yet, so generate one first:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub
```

Add the printed key at <https://github.com/settings/keys>, then:

```bash
cd /Users/junmei/Documents/aiwork/esm-project/confound-audit
git remote add origin git@github.com:junmeiW/confound-audit.git
git push -u origin main
```

---

## 2b. Repository metadata

Paste this into the **About** field (268 characters; the limit is 350):

```
Detect input-composition confounds in mutation-induced embedding analyses before reading them as model findings. Zero-parameter baselines, stratified permutation nulls and substitution-type matching for protein language model representations. No GPU, no model weights.
```

**Topics** (the tags shown under the description) — add via the gear icon next
to About:

```
protein-language-models  embeddings  confounding  statistics
reproducibility  deep-mutational-scanning  bioinformatics
python  science  research-software
```

**Short one-liner** for places with less room (PyPI summary is already set in
`setup.cfg`; use this variant for Zenodo or talk slides):

```
Zero-parameter confound checks for directional embedding statistics in protein language models.
```

---

## 3. After pushing

### URLs

The repository URL `https://github.com/junmeiW/confound-audit` is already set in
`README.md` (CI badge) and in `SETUP_GITHUB.md`. One place still needs your
attention:

- **The manuscript** (`../MANUSCRIPT.md`, "Reusable diagnostic software"): **done**.
  The repository URL and the Zenodo DOI (`10.5281/zenodo.23253177`) are both in the
  text, and a software entry was added as reference [11].

### Check CI

Open the **Actions** tab. Four jobs should run green:

| Job | What it proves |
|---|---|
| tests (3.8–3.12) | works across supported Pythons |
| minimum dependency versions | the declared floors are real |
| build wheel + clean install | the artifact is correct and the CLI works |
| lint | ruff is clean |

### Enable the DOI (recommended)

GPB accepts a repository URL, but a **DOI** is more durable and is what many
journals prefer. Zenodo is the simplest route:

1. Sign in at <https://zenodo.org> with GitHub.
2. Settings → GitHub → toggle **ON** for `confound-audit`.
3. Back on GitHub, create a **release** (e.g. tag `v0.1.0`). Zenodo archives it
   and mints a DOI automatically.
4. Put that DOI into the manuscript.

**Status: done.** Release `v0.1.0` was archived as
<https://doi.org/10.5281/zenodo.23253177> (concept DOI: it resolves to the latest
version), and is cited as reference [11]. Repeat steps 3–4 for any future release
you cite.

---

## 4. Publishing to PyPI (optional, later)

`pip install confound-audit` only works once the name is registered. Until
then, readers install from GitHub:

```bash
pip install git+https://github.com/junmeiW/confound-audit.git
```

To publish properly:

```bash
cd /Users/junmei/Documents/aiwork/esm-project/confound-audit
python -m pip install --upgrade build twine
rm -rf dist && python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*   # dry run on TestPyPI
python -m twine upload dist/*                          # real upload
```

Check the name is still free first: <https://pypi.org/project/confound-audit/>

---

## Checklist

- [ ] Set `user.name` / `user.email` (not the placeholder)
- [ ] Add `confound-audit/` to the **parent** repo's `.gitignore`
- [ ] `./scripts/init_repo.sh --dry-run`, review, then run for real
- [ ] Confirm 28 files and no `dist/`, `__pycache__/`, `.pytest_cache/`
- [ ] Create the empty GitHub repo (no README/licence)
- [ ] Push, and confirm CI is green
- [x] Insert the repo URL and Zenodo DOI into `MANUSCRIPT.md`
- [x] Mint a Zenodo DOI (`10.5281/zenodo.23253177`) and cite it as reference [11]
