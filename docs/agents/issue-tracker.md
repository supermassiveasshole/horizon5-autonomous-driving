# Issue Tracker: GitHub

Issues and task specifications live in [GitHub Issues](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues). The product baseline remains in [PRD](../design/PRD.md).

## Repository and CLI

Use the `gh` CLI from the repository root; verify the target with `git remote -v`. The configured repository is `supermassiveasshole/horizon5-autonomous-driving`. Supply `--repo` explicitly when operating outside the checkout.

On this Windows setup, if `gh` is absent from PATH, invoke `& '.\.tools\github-cli\bin\gh.exe'` in place of `gh`. The ignored `.tools/` directory is local tooling; other contributors install GitHub CLI separately. Use an authenticated session; keep credentials out of repository files, command arguments, and output.

## Issue Operations

Write multiline issue bodies and comments to UTF-8 files under the ignored `.scratch/` directory, then use `--body-file`. Keep the complete text, including Markdown and newlines, in that file.

```powershell
# Run from the repository root; replace 42 with the intended issue number.
gh issue create --title 'Describe the task' --body-file .scratch/issue-body.md --label needs-triage
gh issue view 42 --comments
gh issue view 42 --json number,title,body,labels,comments,state,assignees
gh issue list --state open --json number,title,labels,assignees
gh issue comment 42 --body-file .scratch/issue-comment.md
gh issue edit 42 --add-label ready-for-agent --remove-label needs-triage
gh issue close 42
```

Read the issue, labels, and discussion before changing it. Post the resolution and validation evidence before closing. For longer issue inventories, paginate or increase the limit explicitly.

When a skill says **publish to the issue tracker**, create a GitHub issue. When it says **fetch the relevant ticket**, read that issue and its comments. Use the mappings in [triage-labels.md](triage-labels.md).

## Pull Requests as a Triage Surface

**PRs as a request surface: no.** Ordinary development and review PRs remain supported. GitHub shares numbering between issues and PRs; resolve an ambiguous `#42` with `gh pr view 42`, falling back to `gh issue view 42` when it is an issue. Handle authentication and network errors before making that distinction.

## Wayfinding

Apply this section when the `wayfinder` skill is invoked.

- Keep the map in one issue labelled `wayfinder:map`, with Notes, Decisions-so-far, and Fog sections.
- Create child tickets as GitHub sub-issues when supported; otherwise maintain a task list in the map and a `Part of #<map>` line in each child. Use `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task` for the corresponding ticket type; create missing labels when this workflow is used.
- Record blockers using native GitHub issue dependencies when available. If using their REST endpoint, use the blocker's numeric database ID rather than its issue number or node ID. Otherwise record `Blocked by: #<number>` in the child.
- Select the first open child in map order with no assignee and no open blockers. Scope the search to this map's children and check each blocker.
- Claim the ticket with `gh issue edit 42 --add-assignee '@me'` before implementation.
- Resolve with an evidence-bearing comment via `--body-file`, close the child, and add a short decision plus its issue link to the map.
