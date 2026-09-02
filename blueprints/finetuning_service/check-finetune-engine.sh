#!/bin/bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

CONFIG_FILE="$(dirname "$(realpath "$0")")/finetune-config.cfg"

# Load values from finetune-config.cfg (strip quotes and inline comments)
get_cfg() { grep -E "^${1}:" "$CONFIG_FILE" 2>/dev/null | sed -E "s/^${1}:[[:space:]]*//;s/[[:space:]]*#.*$//;s/^\"//;s/\"$//;s/^'//;s/'$//"; }

BACKEND=$(get_cfg nvidia_finetune_backend_url)
TOKEN_URL=$(get_cfg nvidia_keycloak_token_url)
CID=$(get_cfg nvidia_keycloak_client_id)
CSEC=$(get_cfg nvidia_keycloak_client_secret)

echo "Config file: $CONFIG_FILE"
echo "Backend URL: ${BACKEND:-<not set>}"
echo "Token URL:   ${TOKEN_URL:-<not set>}"
echo

# Step 0: config completeness
if [[ -z "$BACKEND" || -z "$TOKEN_URL" || -z "$CID" || -z "$CSEC" ]]; then
  echo "ACTION: One or more values are missing in finetune-config.cfg."
  echo "        Please edit $CONFIG_FILE and set:"
  echo "          nvidia_finetune_backend_url, nvidia_keycloak_token_url,"
  echo "          nvidia_keycloak_client_id, nvidia_keycloak_client_secret"
  exit 1
fi

status() { [[ "$1" == "200" ]] && echo "OK ($1)" || echo "FAIL ($1)"; }

CODE=$(curl -sS -o /dev/null -w '%{http_code}' "$BACKEND/docs" --max-time 5 2>/dev/null)
echo "API health:  $(status $CODE)"

TOK_RESP=$(curl -sS -X POST "$TOKEN_URL" -d "grant_type=client_credentials&client_id=$CID&client_secret=$CSEC" --max-time 5 2>/dev/null)
TOKEN=$(echo "$TOK_RESP" | python3 -c "import json,sys;print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)
[[ -n "$TOKEN" ]] && echo "Token:       OK" || echo "Token:       FAIL"

if [[ -n "$TOKEN" ]]; then
  CODE=$(curl -sS -o /dev/null -w '%{http_code}' "$BACKEND/finetune/jobs" -H "Authorization: Bearer $TOKEN" --max-time 5 2>/dev/null)
  echo "Auth call:   $(status $CODE)"
else
  echo "Auth call:   SKIPPED (no token)"
fi

# Step: hint based on results
echo
if [[ "$CODE" == "200" && -n "$TOKEN" ]]; then
  echo "STATUS: Fine-tuning engine is reachable and authenticating successfully."
else
  ENGINE_HOST=$(echo "$BACKEND" | sed -E 's~^https?://([^:/]+).*~\1~')
  echo "ACTION: The fine-tuning engine is not responding."
  echo "        1. Verify the URLs in $CONFIG_FILE are correct."
  echo "        2. Log in to the engine host ($ENGINE_HOST) and start the fine-tuning engine."
fi
