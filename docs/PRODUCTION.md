# Production Deployment Guide

This guide covers deploying the FL server to a public cloud VM and connecting remote manufacturing sites over the internet.

---

## Architecture

```
Internet
   │
   │  HTTPS (443 or 8000)
   │
┌──▼──────────────────────────────┐
│  Cloud VM / On-prem Server      │
│                                 │
│  nginx (optional TLS terminator)│
│    └─► FastAPI on port 8000     │
│    └─► Flet dashboard port 8550 │
└─────────────────────────────────┘
   │          │          │
   │          │          │
 site_1     site_2     site_3 …   (separate servers / networks)
(Python FL client, connects out)
```

Sites **never** receive inbound connections — they connect outbound to the server only. No inbound firewall rules needed on site machines.

---

## Step 1 — Provision the server

Any Linux VM with Docker works:

```bash
# Ubuntu 22.04 example
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
newgrp docker
```

Minimum recommended spec: 2 vCPU, 4 GB RAM, 20 GB disk.

Ports to open in your cloud firewall / security group:
| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 443  | TCP      | inbound   | HTTPS (if using nginx TLS termination) |
| 8000 | TCP      | inbound   | FastAPI (if exposing directly with TLS) |
| 8550 | TCP      | inbound   | Flet dashboard (restrict to admin IPs) |
| 22   | TCP      | inbound   | SSH (restrict to your IPs) |

---

## Step 2 — TLS certificate

### Option A — Let's Encrypt (recommended, free)

```bash
sudo apt install -y certbot
sudo certbot certonly --standalone -d fl-server.yourdomain.com
# Cert: /etc/letsencrypt/live/fl-server.yourdomain.com/fullchain.pem
# Key:  /etc/letsencrypt/live/fl-server.yourdomain.com/privkey.pem
```

### Option B — Nginx TLS termination (if you prefer a reverse proxy)

```nginx
# /etc/nginx/sites-available/viral_fl
server {
    listen 443 ssl;
    server_name fl-server.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/fl-server.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fl-server.yourdomain.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # FL REST API
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/viral_fl /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Option C — Self-signed (dev/staging only)

```bash
bash scripts/generate_certs.sh
# Then set SSL_KEYFILE and SSL_CERTFILE in .env (see below)
# Set VERIFY_SSL=false on each client
```

---

## Step 3 — Configure `.env` on the server

```bash
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning
cp .env.example .env
```

Edit `.env`:

```ini
# Secrets
SERVER_SECRET_KEY=<64-char random hex>  # python -c "import secrets; print(secrets.token_hex(32))"
SITE_1_SECRET=<random>
SITE_2_SECRET=<random>
SITE_3_SECRET=<random>
SITE_4_SECRET=<random>
SITE_5_SECRET=<random>

# Database (Docker Compose handles this automatically)
SERVER_DB_URL=postgresql+asyncpg://viral_fl:viral_fl_pass@db:5432/viral_fl

# TLS — Option A / C (skip if using nginx Option B)
SSL_KEYFILE=/certs/server.key
SSL_CERTFILE=/certs/server.crt

# CORS — list all site dashboard origins
CORS_ORIGINS=https://site1.company.com:8551,https://site2.company.com:8552,https://site3.company.com:8553,https://site4.company.com:8554,https://site5.company.com:8555
```

Start the server:

```bash
docker compose up -d db server
# Check logs
docker compose logs -f server
```

---

## Step 4 — Configure each remote site

On each site's server (separate machine, any OS with Python 3.12+):

```bash
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/client.txt

cp .env.example .env
```

Edit `.env` on each site (example for site_1):

```ini
SITE_ID=site_1
SERVER_URL=https://fl-server.yourdomain.com   # or http://IP:8000 if no TLS
SITE_SECRET=<same value as SITE_1_SECRET on server>
VERIFY_SSL=true                               # false only for self-signed certs
CONNECT_TIMEOUT=15
REQUEST_TIMEOUT=90
RETRY_ATTEMPTS=5

LOCAL_DATA_PATH=/path/to/local/filtration.csv
FLET_CLIENT_PORT=8551
```

Generate synthetic data (first run only):

```bash
python scripts/generate_synthetic_data.py
```

Start the client:

```bash
python -m client.main
```

The client connects outbound to `SERVER_URL`. No inbound ports need to be open on the site machine.

---

## Step 5 — Firewall / network requirements per site

| Requirement | Detail |
|-------------|--------|
| Outbound HTTPS to server | TCP port 443 (nginx) or 8000 (direct). Standard outbound — most corporate firewalls allow this. |
| No inbound ports needed | Clients only make outbound connections. |
| Raw data stays local | Only gradient updates leave the site (Gaussian DP noise applied first). |
| Proxy support | If the site uses an HTTP proxy, set `HTTP_PROXY` / `HTTPS_PROXY` environment variables; `httpx` picks these up automatically. |

---

## Environment variable differences: dev vs prod

| Variable | Dev (Docker) | Production |
|----------|-------------|------------|
| `SERVER_URL` | `http://server:8000` | `https://fl-server.yourdomain.com` |
| `VERIFY_SSL` | `false` (self-signed) or not set | `true` |
| `CORS_ORIGINS` | empty (allow all, dev only) | comma-separated site origins |
| `SSL_KEYFILE` / `SSL_CERTFILE` | optional | recommended (or use nginx) |
| `RETRY_ATTEMPTS` | `3` | `5` (internet has higher latency/loss) |
| `REQUEST_TIMEOUT` | `60` | `90`–`120` |

---

## Monitoring

- **Server dashboard**: `https://fl-server.yourdomain.com:8550` — round progress, site status, global model convergence.
- **API health**: `GET https://fl-server.yourdomain.com/health/` — JSON `{"status": "ok"}`.
- **Logs**: `docker compose logs -f server` on the server host.

---

## Security checklist before going live

- [ ] `SERVER_SECRET_KEY` is a strong random secret (not `CHANGE_ME`)
- [ ] All `SITE_N_SECRET` values are unique, strong, and securely distributed to sites
- [ ] TLS enabled on the server endpoint
- [ ] `CORS_ORIGINS` set to specific site origins (not empty)
- [ ] PostgreSQL password changed from default `viral_fl_pass`
- [ ] Flet dashboard port 8550 restricted to admin IPs in cloud firewall
- [ ] SSH key-only authentication on the server VM
- [ ] `DP_NOISE_SIGMA` set appropriately for your privacy budget (≥ 0.01)
