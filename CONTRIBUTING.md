# How we work

## One-time setup (each teammate)

```bash
git clone https://github.com/<org-or-user>/Parallax.git
cd Parallax
git config user.name "Your Name"
git config user.email "your@email.com"
```

## Every feature, every time

```bash
git checkout main
git pull origin main              # always start from the latest main
git checkout -b feat/ocr-pdf-ingest

# ...write code...

git add -A
git commit -m "feat(ocr): extract text from multi-page PDFs"
git push -u origin feat/ocr-pdf-ingest
```

Then open a Pull Request on GitHub (or `gh pr create --fill`), get one teammate to review, and
**Squash and merge**. Delete the branch after merging.

## Branch naming

| Prefix | Use for | Example |
| --- | --- | --- |
| `feat/` | new feature | `feat/teachback-scoring` |
| `fix/` | bug fix | `fix/rag-empty-chunk-crash` |
| `chore/` | deps, config, cleanup | `chore/add-ruff` |
| `docs/` | docs only | `docs/demo-script` |

Keep the area in the name (`ocr`, `rag`, `tutor`, `teachback`, `practice`, `ui`) so anyone can tell
at a glance whose branch it is.

## Commit messages

`type(area): what changed` — e.g. `feat(rag): reject queries below similarity threshold`.
Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.

## Rules that save us during the demo

1. **Never push to `main` directly.** Every change goes through a PR.
2. **Never commit secrets.** API keys live in `.env`, which is gitignored. Commit `.env.example` with
   empty placeholder keys instead.
3. **Never commit large binaries** — model weights, vector indexes, sample PDFs over a few MB.
4. **Pull before you branch, rebase before you push.** If `main` moved while you worked:
   ```bash
   git checkout main && git pull origin main
   git checkout feat/your-branch
   git rebase main
   # fix any conflicts, then:
   git push --force-with-lease
   ```
5. **Small PRs.** One feature per PR beats one giant PR the night before judging.
6. **`main` must always run.** If it's broken, that's the top priority — nothing else merges until it's fixed.

## Resolving a conflict

Git marks conflicts inside the file with `<<<<<<<`, `=======`, `>>>>>>>`. Edit the file so it contains
the version you want, delete the markers, then:

```bash
git add <file>
git rebase --continue     # or: git commit, if you were merging
```

If it gets messy: `git rebase --abort` puts you back where you started. Nothing is lost.
