#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

target="${1:-descriptors}"

case "$target" in
  descriptors)
    app_path="apps/descriptors_app.py"
    ;;
  similarity)
    app_path="apps/similarity_app.py"
    ;;
  text)
    app_path="apps/text_query_app.py"
    ;;
  *)
    app_path="$target"
    ;;
esac

if [ -x ".venv/bin/streamlit" ]; then
  streamlit_cmd=".venv/bin/streamlit"
else
  streamlit_cmd="streamlit"
fi

"$streamlit_cmd" run "$app_path"
