#!/usr/bin/env bash
# Install the three closed-loop suites for `jevarena-mm play` (docs/envs.md): each upstream at the revision of
# data/loop/suites.json, in its own Python as its README asks (GameWorld 3.12, BALROG and VS-Bench 3.10, made by uv),
# then print the .env lines that point jevarena-mm at them.
#
#   scripts/loop/install.sh /path/to/suites
#
# Needs git, network access, and uv or a python3 to install it into DIR. Takes about 10 minutes and 3 GB. Running it
# again skips what is installed. GameWorld's Chromium needs the usual browser libraries of the system
# (`playwright install-deps chromium` installs them as root).
set -euo pipefail
DIR=$(realpath -m "${1:?usage: scripts/loop/install.sh DIR}")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GAMEWORLD=3c26bdab436800fd61ef40543b64ca40d12c7e4a
GAMEWORLD_GAMES=55322928fa8bd51cb1719bd3807a32634aa5d3cb  # GameWorld does not pin its game library; 2026-09-01
BALROG=b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5
VSBENCH=2c3531cdf91d1b620b4a9015126a242eb305a4d0

mkdir -p "$DIR"
export UV_PYTHON_INSTALL_DIR="$DIR/python"
if command -v uv >/dev/null; then
  UV=uv
else
  [ -x "$DIR/uv/bin/uv" ] || { python3 -m venv "$DIR/uv" && "$DIR/uv/bin/pip" install -q uv; }
  UV="$DIR/uv/bin/uv"
fi
checkout() {  # checkout DIR URL REVISION
  [ -d "$1/.git" ] || git clone -q "$2" "$1"
  git -C "$1" checkout -q "$3" 2>/dev/null || { git -C "$1" fetch -q origin && git -C "$1" checkout -q "$3"; }
  echo "$(basename "$1") @ $3"
}
venv() {  # venv DIR PYTHON_VERSION
  [ -x "$1/bin/python" ] || "$UV" venv -q -p "$2" "$1"
}

checkout "$DIR/GameWorld" https://github.com/gameworld-project/gameworld.git "$GAMEWORLD"
checkout "$DIR/GameWorld/games/gameworld-games" https://github.com/gameworld-project/GameWorld-Games.git "$GAMEWORLD_GAMES"
venv "$DIR/venv-gameworld" 3.12
(cd "$DIR/GameWorld" && "$UV" pip install -q -p "$DIR/venv-gameworld/bin/python" -r requirements.txt)
PLAYWRIGHT_BROWSERS_PATH="$DIR/browsers" "$DIR/venv-gameworld/bin/playwright" install chromium

checkout "$DIR/BALROG" https://github.com/balrog-ai/BALROG.git "$BALROG"
venv "$DIR/venv-balrog" 3.10
# BALROG pins neither Gymnasium nor NumPy: hold them where its wrappers and gym 0.23 work (balrog-constraints.txt).
# Once installed, only those are checked: resolving BALROG's git dependencies again can stall on GitHub.
if "$DIR/venv-balrog/bin/python" -c "import balrog" 2>/dev/null; then
  "$UV" pip install -q -p "$DIR/venv-balrog/bin/python" -c "$HERE/balrog-constraints.txt" gymnasium numpy setuptools
else
  "$UV" pip install -q -p "$DIR/venv-balrog/bin/python" -c "$HERE/balrog-constraints.txt" -e "$DIR/BALROG" setuptools
fi
if [ ! -e "$DIR/BALROG/.post-install-done" ]; then  # MiniHack's Boxoban levels and the TextWorld games
  (cd "$DIR/BALROG" && PATH="$DIR/venv-balrog/bin:$PATH" balrog-post-install) && touch "$DIR/BALROG/.post-install-done"
fi

checkout "$DIR/VS-Bench" https://github.com/zelaix/VS-Bench.git "$VSBENCH"
venv "$DIR/venv-vsbench" 3.10
(cd "$DIR/VS-Bench" && "$UV" pip install -q -p "$DIR/venv-vsbench/bin/python" -r requirements.txt)

cat <<EOF

# Add to .env (or export):
JEVARENA_MM_GAMEWORLD=$DIR/GameWorld
JEVARENA_MM_GAMEWORLD_PYTHON=$DIR/venv-gameworld/bin/python
JEVARENA_MM_GAMEWORLD_BROWSERS=$DIR/browsers
JEVARENA_MM_BALROG=$DIR/BALROG
JEVARENA_MM_BALROG_PYTHON=$DIR/venv-balrog/bin/python
JEVARENA_MM_VSBENCH=$DIR/VS-Bench
JEVARENA_MM_VSBENCH_PYTHON=$DIR/venv-vsbench/bin/python
EOF
