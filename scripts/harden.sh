#!/usr/bin/env bash
# Lockout-SAFE first-login hardening for a fresh Ubuntu Droplet (run as root):
#   bash harden.sh "<YOUR_LAPTOP_SSH_PUBLIC_KEY>" [username]
#
# Creates a key-authenticated sudo user and sets ufw + fail2ban + swap + UTC +
# automatic security updates. It does NOT disable password login — that's
# lockdown_ssh.sh, run only AFTER you've confirmed key login works, so you can
# never lock yourself out.
set -euo pipefail

PUBKEY="${1:?usage: bash harden.sh \"<ssh-public-key>\" [username]}"
USER_NAME="${2:-trader}"
[[ $EUID -eq 0 ]] || { echo "Run as root."; exit 1; }
case "$PUBKEY" in
  ssh-ed25519*|ssh-rsa*|ecdsa-sha2-*) ;;
  *) echo "Arg 1 must be an SSH PUBLIC key (starts with 'ssh-ed25519 ...')."; exit 1 ;;
esac
export DEBIAN_FRONTEND=noninteractive

id "$USER_NAME" &>/dev/null || adduser --disabled-password --gecos "" "$USER_NAME"
usermod -aG sudo "$USER_NAME"
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.ssh"
printf '%s\n' "$PUBKEY" >"/home/$USER_NAME/.ssh/authorized_keys"
chmod 600 "/home/$USER_NAME/.ssh/authorized_keys"
chown "$USER_NAME:$USER_NAME" "/home/$USER_NAME/.ssh/authorized_keys"

# Passwordless sudo for the key-authenticated admin (single-admin box; the SSH
# key is the gate). Swap to a sudo password later if you prefer.
echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >"/etc/sudoers.d/90-$USER_NAME"
chmod 440 "/etc/sudoers.d/90-$USER_NAME"
visudo -cf "/etc/sudoers.d/90-$USER_NAME"

apt-get update -y
apt-get install -y ufw fail2ban unattended-upgrades
timedatectl set-timezone UTC || true

if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi

ufw allow OpenSSH || ufw allow 22/tcp
ufw --force enable
systemctl enable --now fail2ban
dpkg-reconfigure -f noninteractive -plow unattended-upgrades || true

echo "OK: user '$USER_NAME' (key login + passwordless sudo); ufw + fail2ban on; swap + UTC + auto-updates set."
echo "TEST (new local terminal, keep this one open): ssh $USER_NAME@<DROPLET_IP> && sudo whoami"
echo "Then lock down SSH: sudo bash lockdown_ssh.sh $USER_NAME"
