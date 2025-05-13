#!/bin/zsh
echo "Shell: $SHELL"
echo "PATH: $PATH" | grep -o "[^:]*node[^:]*"
echo "Node version: $(node --version)"
echo "Which node: $(which node)"
