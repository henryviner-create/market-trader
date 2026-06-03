#!/bin/bash
# DigitalOcean user-data (runs as root on FIRST BOOT via cloud-init).
# Self-provisions the market-trader PAPER stack with zero further SSH needed.
#
# Placeholders __SSH_PUBKEY__ and __REPO_URL__ are filled in by
# scripts/provision_do.py at create time. NO application secrets live here — the
# paper stack runs without them; add real keys later in /opt/market-trader/.env
# over your own SSH. If __REPO_URL__ embeds a read-only token, revoke it after.
set -eux
exec >/var/log/market-trader-init.log 2>&1  # boot log for debugging

USER_NAME=trader
PUBKEY='__SSH_PUBKEY__'
REPO_URL='__REPO_URL__'
APP_DIR=/opt/market-trader
export DEBIAN_FRONTEND=noninteractive

timedatectl set-timezone UTC || true

# admin user with key login + passwordless sudo (key is the gate)
id "$USER_NAME" &>/dev/null || adduser --disabled-password --gecos "" "$USER_NAME"
usermod -aG sudo "$USER_NAME"
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.ssh"
printf '%s\n' "$PUBKEY" >"/home/$USER_NAME/.ssh/authorized_keys"
chmod 600 "/home/$USER_NAME/.ssh/authorized_keys"
chown "$USER_NAME:$USER_NAME" "/home/$USER_NAME/.ssh/authorized_keys"
echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >"/etc/sudoers.d/90-$USER_NAME"
chmod 440 "/etc/sudoers.d/90-$USER_NAME"

# swap + packages + firewall + fail2ban + auto-updates
if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi
apt-get update -y
apt-get install -y ca-certificates curl git ufw fail2ban unattended-upgrades
ufw allow OpenSSH; ufw allow 80/tcp; ufw allow 443/tcp; ufw --force enable
systemctl enable --now fail2ban
dpkg-reconfigure -f noninteractive -plow unattended-upgrades || true

# key-only SSH from the start (no password, no root login)
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
mkdir -p /etc/ssh/sshd_config.d
printf 'PasswordAuthentication no\nPermitRootLogin no\n' >/etc/ssh/sshd_config.d/99-market-trader.conf
systemctl restart ssh || systemctl restart sshd || true

# Docker
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# App: clone, paper .env, systemd unit, bring the stack up
git clone "$REPO_URL" "$APP_DIR"
git -C "$APP_DIR" remote set-url origin https://github.com/henryviner-create/market-trader.git  # drop any token from the remote
chown -R "$USER_NAME:$USER_NAME" "$APP_DIR"
cp "$APP_DIR/.env.example" "$APP_DIR/.env"  # paper defaults; real keys added later over SSH
cp "$APP_DIR/deploy/market-trader.service" /etc/systemd/system/market-trader.service
systemctl daemon-reload
systemctl enable market-trader.service
systemctl start market-trader.service
echo "market-trader cloud-init complete"
