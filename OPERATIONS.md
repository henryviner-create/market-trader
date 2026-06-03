# Operations runbook

Plain-English operations for running market-trader **paper-first** on a single
Linux VPS. Every command is copy-pasteable. Run them from the repo directory on
the VPS (`/opt/market-trader`).

> **Maturity note.** Phase 0 baseline + Phase 7 hardening are in place: a
> `/metrics` endpoint, data-freshness + heartbeat monitoring, a `monitoring`
> compose profile (Prometheus + Grafana, localhost-bound), and off-box
> backup/restore scripts (`scripts/backup_db.sh`, `scripts/restore_db.sh`). Start
> monitoring with `docker compose --profile monitoring up -d` (Grafana on
> localhost:3000). Alertmanager routing rules are the remaining polish.

## Topology

```
[ db (Postgres+pgvector) ] <— [ migrate (run once) ]   [ engine (serve, health-checked) ]
        named volume pgdata                                    restart: unless-stopped
```

All services run via `docker compose`. The engine never begins trading until the
database is healthy and migrations have completed.

## First-time setup (fresh Droplet)

Target box: a **DigitalOcean Basic Droplet** (4 vCPU / 8 GB / NVMe, Premium
AMD/Intel preferred) running **Ubuntu 24.04 LTS** (any Debian/Ubuntu works).
Enable **two backup layers**: DigitalOcean automated backups (weekly,
whole-server rollback) *and* the off-box `pg_dump` -> Spaces/S3 dumps below
(granular data recovery). Enable DO's metrics agent + Monitoring.

```bash
# As root on a fresh Ubuntu 24.04 box:
git clone <your-repo-url> /opt/market-trader
cd /opt/market-trader
sudo bash scripts/bootstrap_vps.sh      # UTC, swap, updates, Docker, firewall, fail2ban, systemd, paper stack
nano .env                               # DB password + Anthropic/Alpaca PAPER keys; leave live DISABLED
sudo systemctl restart market-trader
```

`bootstrap_vps.sh` is idempotent and defaults to **paper**. It never enables live
trading.

> **Prefer a guided, click-by-click version?** Ask Claude for the **staged
> DigitalOcean provisioning walkthrough** — it covers creating the Droplet, SSH
> keys, first-login hardening, secrets, bring-up, reboot/auto-recovery proof, and
> backups (incl. off-box to Spaces), one confirmable stage at a time, assuming no
> prior server experience.

## Deploy / redeploy

```bash
cd /opt/market-trader
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Migrations run automatically (the `migrate` service) before the engine starts.
**Always back up the database before a migration** (see Backups).

## Start / stop / restart

```bash
# Whole stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d      # start
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop       # stop
sudo systemctl restart market-trader                                       # restart via systemd

# A single service
docker compose restart engine
docker compose restart db
```

A server reboot brings everything back automatically (systemd unit
`market-trader.service` + `restart: unless-stopped`).

## Read logs / find an error

```bash
docker compose logs -f engine            # follow the engine
docker compose logs --since 1h engine    # last hour
docker compose logs engine | grep -i error
docker compose ps                        # is anything unhealthy / restarting?
curl -s localhost:8080/health            # {"status":"ok","db":true,...}
```

Logs are structured JSON in production (`MT_JSON_LOGS=true`) so they are
greppable and ship cleanly to a log aggregator (Phase 7).

## The kill switch

**Today:** stop the engine — it ceases all activity immediately and, on SIGTERM,
stands down cleanly (it does not fire blind):

```bash
docker compose stop engine
```

**Phase 8 (execution tier):** a first-class kill switch command that also cancels
resting orders / flattens positions per policy, plus automatic triggers
(drawdown breach, heartbeat loss). Until live trading exists there is nothing to
flatten — stopping the engine is sufficient.

## Alerts — meaning and response

> The alerting stack (Alertmanager + external heartbeat) lands in Phase 7. The
> table below is the response policy; today you check these manually via
> `docker compose logs` / `/health`.

| Alert | Means | Do this |
|------|-------|---------|
| Heartbeat lost | The box or engine is fully dead | SSH in; `docker compose ps`; `up -d`; check host (disk/mem) |
| Data feed silent (e.g. "EDGAR quiet 6h") | A collector stopped producing | Check that collector's logs; verify the upstream API; re-run the job |
| Job failed | A scheduled job errored | Read its logs; fix input/credentials; re-run (jobs are idempotent) |
| Drift | Live performance diverging from backtest | **Investigate before trusting signals**; consider standing down |
| Guardrail trip | A risk limit was hit | Expected protection — review *why*; do not raise the limit reflexively |
| Disk / CPU / memory | Host resource pressure | Free space (prune images, rotate logs); resize the VPS if persistent |

## Backups & restore

> **Phase 7** automates scheduled, off-box backups with retention + restore
> drills. Use these manual steps until then. **A backup you have not restored is
> not a backup.**

```bash
# Back up (run before every migration). Writes a compressed dump you should copy OFF-box.
docker compose exec -T db pg_dump -U market market_trader | gzip > backup_$(date +%F).sql.gz
# Copy it somewhere off the server (e.g. your laptop / object storage):
#   scp user@vps:/opt/market-trader/backup_*.sql.gz .

# Restore into a fresh database (DESTRUCTIVE — overwrites current data):
gunzip -c backup_YYYY-MM-DD.sql.gz | docker compose exec -T db psql -U market market_trader
```

## Incident response — the first 10 minutes

1. **Breathe.** In paper mode nothing is at financial risk; act deliberately.
2. **Scope it:** `docker compose ps` — what is down/unhealthy/restarting?
3. **Health:** `curl -s localhost:8080/health` — DB reachable?
4. **Logs:** `docker compose logs --since 30m engine db` — find the first error.
5. **Host:** `df -h` and `free -m` — out of disk or memory?
6. **Stabilise:** if the engine is crash-looping, `docker compose stop engine`
   (it stands down cleanly) and investigate before restarting.
7. **DB suspect?** Back up first (above), *then* touch it.
8. **Recover:** fix the cause; `up -d --build`; confirm `/health` is `ok`.
9. **Record:** note what happened and the fix (feeds the feedback loop).
10. **If unsure, stay down.** A stopped paper engine costs nothing; a confused
    running one costs trust.

## When X breaks, do Y

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `engine` keeps restarting | Bad config / DB unreachable | `docker compose logs engine`; check `.env` `MT_DATABASE_URL`; ensure `db` healthy |
| `/health` returns 503 | DB down or migration pending | `docker compose ps db`; re-run `docker compose run --rm migrate` |
| "port already in use" | Stale container / another process | `docker compose down`; `docker ps`; remove the conflict |
| Disk full | Image/log buildup | `docker system prune -af`; rotate/ship logs; resize VPS |
| Migration failed | Schema/data conflict | Restore the pre-migration backup; investigate; retry |
| Can't SSH in | Firewall / key issue | Use the provider's web console; check `ufw status`, `sshd` |

## Environments

| Env | Host | Mode | Notes |
|-----|------|------|-------|
| dev | laptop | paper | `docker compose up`; mock/sandbox data fine |
| paper (staging) | the VPS | paper | the long-running experiment — the system's normal home |
| live | separate, **later, gated** | live | tiny capital ceiling, all guardrails on; parity with paper except mode/keys/cap |

Live is never enabled without the paper→live graduation gates met **and** an
explicit human decision (see `DECISIONS.md` D10).
