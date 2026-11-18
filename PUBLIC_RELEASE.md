# Public release checklist

Complete this review before pushing a branch to the public repository. A push
can expose every new commit, including content removed by a later commit.

- Confirm each tracked addition serves the public source, tests, documentation,
  or an explicitly sanitized configuration example.
- Keep `research/`, original source documents, downloaded files, datasets,
  extracts, catalog manifests, and private evaluation results out of the index.
  Confirm synthetic fixtures contain invented content.
- Exclude model weights, checkpoints, optimizer state, and run directories.
  Re-run the local baseline verifier when touching code around legacy artifacts.
- Exclude raw rights evidence, license correspondence, approval and revocation
  records, reviewer identities, and private operator configuration. Public
  attribution or example notices must have independently cleared scope.
- Scan tracked contents and every outgoing commit for secrets or private
  identifiers. Check credentials, tokens, embedded authentication in URLs,
  absolute private paths, and environment files. Verify author and committer
  metadata uses a publishing identity and an intended public email address.
- Stage explicit paths. Review `git status --short`, `git diff --cached --check`,
  `git diff --cached --name-only`, and the full `git diff --cached`. Inspect
  `git log --stat origin/main..HEAD` once committed. Use `git check-ignore -v`
  on representative private paths and `git ls-files -ci --exclude-standard`
  to find already tracked files that match ignore rules.
- Confirm the clean-checkout CPU tests pass without access to research data
  or weights. Push a feature branch, open a pull request, and inspect its files
  and checks before review or merge.

If a credential was published, revoke or rotate it immediately. Removing it
from the current tree leaves previous public commits and copies accessible.
