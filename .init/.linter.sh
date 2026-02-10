#!/bin/bash
cd /home/kavia/workspace/code-generation/battleships-web-game-216474-216483/battleships_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

