#!/usr/bin/env bash
set -euo pipefail

# List of required TeX Live packages
required_pkgs=(
  dvipng
  cm-super
  fix-cm
  collection-latexextra
  collection-fontsrecommended
)

# 1) Do we already have a TeX installation? Prefer tlmgr if present.
if ! command -v tlmgr >/dev/null 2>&1; then
  echo "✗ tlmgr not found. Installing TinyTeX…"
  wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
  # Ensure tlmgr is now on our PATH
  export PATH="$HOME/.TinyTeX/bin/$(uname -m)-*-linux:$PATH"
fi

# 2) Check for missing packages
missing=()
for pkg in "${required_pkgs[@]}"; do
  # `installed: Yes` only if already present
  if ! tlmgr info "$pkg" | grep -q "installed: Yes"; then
    missing+=("$pkg")
  fi
done

# 3) Install any that weren’t found
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing TeX packages: ${missing[*]}"
  echo "Installing missing packages via tlmgr..."
  tlmgr install "${missing[@]}"
else
  echo "All required TeX packages are already installed."
fi
