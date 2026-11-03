#!/usr/bin/env bash
# PreToolUse hook for the Bash tool: blocks obviously destructive commands
# before they hit the shell. Doesn't replace judgment, it just catches the
# clearly dangerous case (a loose rm -rf, force-push, reset --hard, etc).

set -euo pipefail

input="$(cat)"

command="$(printf '%s' "$input" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
")"

if [ -z "$command" ]; then
  exit 0
fi

deny() {
  echo "Blocked by validate-bash.sh hook: $1" >&2
  exit 2
}

# git merge without --no-ff: checked separately because "flag is absent"
# doesn't translate into a simple case/glob pattern.
if printf '%s' "$command" | grep -qE '(^|[;&|]|\s)git merge(\s|$)' \
   && ! printf '%s' "$command" | grep -qE -- '--no-ff'; then
  deny "git merge without --no-ff. This repo preserves branch history: use 'git merge --no-ff <branch>'."
fi

case "$command" in
  *"rm -rf /"*|*"rm -rf ~"*|*"rm -rf ."*|*"rm -fr /"*)
    deny "rm -rf on a path that's too broad. Run it manually if that's really the intent."
    ;;
  *"git push --force"*|*"git push -f "*|*"git push -f\"*"*)
    deny "force-push. Confirm with the user first."
    ;;
  *"git reset --hard"*)
    deny "git reset --hard discards uncommitted work. Confirm with the user first."
    ;;
  *"git clean -f"*|*"git clean -fd"*|*"git clean -fdx"*)
    deny "git clean -f removes untracked files with no way back. Confirm with the user first."
    ;;
  *"chmod -R 777"*)
    deny "chmod -R 777 is too permissive for normal use."
    ;;
  *"DROP TABLE"*|*"DROP DATABASE"*|*"drop table"*|*"drop database"*)
    deny "DROP on a database. This shouldn't run via an agent without explicit confirmation."
    ;;
  *"git branch -D "*|*"git branch -d "*|*"git branch --delete"*)
    deny "branch delete. This repo never deletes a branch, not even a merged one. Confirm with the user first."
    ;;
  *"git push origin --delete"*|*"git push --delete"*|*"git push origin :"*)
    deny "remote branch delete. This repo never deletes a branch. Confirm with the user first."
    ;;
  *"docker compose down -v"*|*"docker-compose down -v"*)
    deny "tears down the dev Postgres AND deletes the volume (loses all local data). Confirm with the user first."
    ;;
esac

exit 0
