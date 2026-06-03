#!/usr/bin/env bash
#
# One-command bootstrap for a fresh Hetzner Cloud CX32 running Ubuntu 24.04 LTS
# (works on any Debian/Ubuntu box). Idempotent: safe to re-run. Sets up time/UTC,
# swap, security updates, Docker, a firewall, fail2ban, the systemd boot unit, and
# brings the PAPER stack up.
#
# Usage (run as root, from the cloned repo at /opt/market-trader):
#   sudo bash scripts/bootstrap_vps.sh
#
# It NEVER enables live trading. Defaults are paper. Edit .env for secrets.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root (sudo bash scripts/bootstrap_vps.sh)." >&2
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive

log "1/9  System packages"
apt-get update -y
apt-get install -y ca-certificates curl ufw fail2ban git unattended-upgrades

log "2/9  Timezone UTC + time sync (scheduling bugs love timezone ambiguity)"
timedatectl set-timezone UTC || true
systemctl enable --now systemd-timesyncd || true

log "3/9  Swap (2 GB) — headroom so the box never OOM-kills the engine"
if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
else
  echo "Swap already configured."
fi

log "4/9  Automatic security updates"
dpkg-reconfigure -f noninteractive -plow unattended-upgrades || true

log "5/9  Docker Engine + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
else
  echo "Docker already installed: $(docker --version)"
fi
systemctl enable --now docker

log "6/9  Firewall (ufw): allow SSH + HTTP/HTTPS, deny everything else inbound"
ufw allow OpenSSH || ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status verbose || true

log "7/9  fail2ban (brute-force protection for SSH)"
systemctl enable --now fail2ban

log "8/9  Configuration (.env)"
cd "${APP_DIR}"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — EDIT IT (DB password, Anthropic + Alpaca PAPER keys)."
  echo "Defaults are PAPER and safe; live stays disabled until you earn + approve it."
else
  echo ".env already exists; leaving it untouched."
fi

log "9/9  systemd unit + build and start the PAPER stack"
cp deploy/market-trader.service /etc/systemd/system/market-trader.service
systemctl daemon-reload
systemctl enable market-trader.service
systemctl start market-trader.service
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps || true

cat <<'NEXT'

Bootstrap complete. The paper stack is starting.

Next steps:
  * Edit secrets:   nano /opt/market-trader/.env   (then: systemctl restart market-trader)
  * Status:         docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
  * Engine logs:    docker compose logs -f engine
  * Health:         curl -s localhost:8080/health

Recommended next (do once key login works — do NOT lock yourself out):
  * Create a non-root sudo user and add your SSH public key to it.
  * Then in /etc/ssh/sshd_config set:  PasswordAuthentication no   (and: systemctl reload ssh)

For a guided, click-by-click version of all of this (creating the Hetzner server,
SSH keys, secrets), ask Claude for the staged provisioning walkthrough.
See OPERATIONS.md for the full runbook (deploy, backups, alerts, incident response).
NEXT
