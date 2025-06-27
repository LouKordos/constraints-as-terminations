#!/usr/bin/env bash
set -euo pipefail

# Required TeX packages (tlmgr names) and their Debian metapackage equivalents
declare -A PKG_MAP=(
  [dvipng]=dvipng
  [cm-super]=fonts-cm-super
  [fix-cm]=texlive-latex-base
  [collection-latexextra]=texlive-latex-extra
  [collection-fontsrecommended]=texlive-fonts-recommended
)

# Helper: which distro
is_debian() {
  [[ -r /etc/debian_version ]]
}

# 1) Do we have a working tlmgr?
if command -v tlmgr >/dev/null 2>&1; then
  # Test if tlmgr can install without error
  if ! echo > /dev/null 2>&1 <<<"tlmgr info" | tlmgr info 2>&1 | grep -q 'user mode not initialized'; then
    USE_TLMGR=1
  else
    USE_TLMGR=0
  fi
else
  USE_TLMGR=0
fi

if (( USE_TLMGR )); then
  echo "✓ Using tlmgr to manage TeX Live packages."
  missing=()
  for pkg in "${!PKG_MAP[@]}"; do
    if ! tlmgr info "$pkg" | grep -q "installed: Yes"; then
      missing+=("$pkg")
    fi
  done

  if (( ${#missing[@]} )); then
    echo "→ Installing missing tlmgr packages: ${missing[*]}"
    tlmgr install "${missing[@]}"
  else
    echo "✓ All tlmgr-managed TeX packages are already installed."
  fi

else
  echo "⚠️  tlmgr unavailable or disabled (Debian/Ubuntu style). Falling back to apt-get."

  # Build list of Debian metapackages to install
  install_list=()
  for pkg in "${!PKG_MAP[@]}"; do
    # We treat the Debian package as “installed” if dpkg knows about it
    dpkg -s "${PKG_MAP[$pkg]}" &>/dev/null || install_list+=("${PKG_MAP[$pkg]}")
  done

  if (( ${#install_list[@]} )); then
    echo "→ Installing via apt: ${install_list[*]}"
    sudo apt-get update
    sudo apt-get install -y "${install_list[@]}"
  else
    echo "✓ All Debian-packaged TeX dependencies are already installed."
  fi
fi
