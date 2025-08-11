#!/usr/bin/env bash
set -e

# Minimal remote runner: rsync, run command, rsync back new files
# Usage:
#   ./scripts/cloudrun.sh -- <command to run remotely>

REMOTE_HOST="debugpod"
REMOTE_DIR="~/cs336-assignment2-systems"

# Require -- then a command
if [[ "$1" != "--" ]]; then
  echo "Usage: $0 -- <command>" >&2
  exit 1
fi
shift
if [[ $# -eq 0 ]]; then
  echo "No command provided." >&2
  exit 1
fi
REMOTE_CMD="$*"


echo "Syncing to ${REMOTE_HOST}:${REMOTE_DIR}..."
rsync -az --exclude='.git' --exclude='.venv' --exclude='tmp' ./ "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Running remotely: $REMOTE_CMD"
ssh "$REMOTE_HOST" "source ${REMOTE_DIR}/.venv/bin/activate && cd ${REMOTE_DIR} && $REMOTE_CMD"

echo "Done."