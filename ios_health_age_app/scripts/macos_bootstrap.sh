#!/usr/bin/env bash
set -euo pipefail

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "xcodebuild not found. Install Xcode and Command Line Tools first."
  exit 1
fi

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "xcodegen not found. Installing via Homebrew..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install Homebrew, then run: brew install xcodegen"
    exit 1
  fi
  brew install xcodegen
fi

echo "Generating Xcode project..."
xcodegen generate
echo "Done. Open HealthAgeApp.xcodeproj in Xcode."
