#!/usr/bin/env bash
# Disable SSH password + root login. Run as root AFTER you've confirmed key login
# for the new user works (so you cannot lock yourself out):
#   sudo bash lockdown_ssh.sh [username]
#
# Recovery fallback if anything goes wrong: the DigitalOcean web Droplet Console
# (Access -> Launch Droplet Console) bypasses SSH entirely.
set -euo pipefail

USER_NAME="${1:-trader}"
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }

AUTHKEYS="/home/$USER_NAME/.ssh/authorized_keys"
if [[ ! -s "$AUTHKEYS" ]]; then
  echo "REFUSING: $AUTHKEYS is missing/empty — disabling password auth now would lock you out."
  exit 1
fi

sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
# DigitalOcean/cloud-init ships a drop-in that re-enables password auth; override it.
if [[ -d /etc/ssh/sshd_config.d ]]; then
  printf 'PasswordAuthentication no\nPermitRootLogin no\n' >/etc/ssh/sshd_config.d/99-market-trader.conf
fi
systemctl restart ssh 2>/dev/null || systemctl restart sshd
echo "Locked down: SSH password auth + root login disabled."
