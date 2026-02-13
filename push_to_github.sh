#!/bin/bash
# Helper script to sync with GitHub repository

REPO_URL="https://github.com/changjongkim/Cascade-kcj.git"
REMOTE_NAME="origin"

echo "Configuring git remote..."

# Check if remote exists
if git remote | grep -q "$REMOTE_NAME"; then
    CURRENT_URL=$(git remote get-url "$REMOTE_NAME")
    if [[ "$CURRENT_URL" != *"changjongkim/Cascade-kcj"* ]]; then
        echo "Updating remote URL from $CURRENT_URL to $REPO_URL"
        git remote set-url "$REMOTE_NAME" "$REPO_URL"
    else
        echo "Remote '$REMOTE_NAME' is already set to $CURRENT_URL"
    fi
else
    echo "Adding remote '$REMOTE_NAME' -> $REPO_URL"
    git remote add "$REMOTE_NAME" "$REPO_URL"
fi

echo "Pushing to GitHub (main branch)..."
# Try push. If it fails due to auth, user will see the prompt.
git push -u "$REMOTE_NAME" main
