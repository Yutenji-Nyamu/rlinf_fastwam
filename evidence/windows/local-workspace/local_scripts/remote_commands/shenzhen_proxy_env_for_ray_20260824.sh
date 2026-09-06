#!/usr/bin/env bash
set -u
env | grep -iE '^(http|https|all|no)_proxy=' | sort || true
