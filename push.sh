#!/bin/bash

clear

git add .

git status

read -p "Enter commit message: " commit_message

echo "Running: git commit -m $commit_message"

git commit -m "$commit_message"

echo "Running: git push"

git push

echo "Git operation complete!"