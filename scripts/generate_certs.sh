#!/usr/bin/env bash
# generate_certs.sh — generate a self-signed TLS certificate for development.
# NOT for production. Use Let's Encrypt or your CA in prod (see docs/PRODUCTION.md).
#
# Usage:
#   bash scripts/generate_certs.sh
#
# Writes:
#   certs/server.key  — private key (chmod 600)
#   certs/server.crt  — self-signed certificate (365 days)
#
# After running, add to .env:
#   SSL_KEYFILE=/certs/server.key
#   SSL_CERTFILE=/certs/server.crt
#   VERIFY_SSL=false   # clients need this when using self-signed cert

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="${SCRIPT_DIR}/../certs"

mkdir -p "$CERTS_DIR"

openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout "${CERTS_DIR}/server.key" \
  -out    "${CERTS_DIR}/server.crt" \
  -subj   "/C=US/ST=Dev/L=Dev/O=ViralFL/OU=Dev/CN=localhost" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost,DNS:server"

chmod 600 "${CERTS_DIR}/server.key"

echo ""
echo "Self-signed certificate written to ${CERTS_DIR}/"
echo ""
echo "Add to your .env (server):"
echo "  SSL_KEYFILE=/certs/server.key"
echo "  SSL_CERTFILE=/certs/server.crt"
echo ""
echo "Add to your .env (each client):"
echo "  VERIFY_SSL=false"
echo "  SERVER_URL=https://localhost:8000"
echo ""
echo "Docker: the server mounts ./certs:/certs:ro automatically."
