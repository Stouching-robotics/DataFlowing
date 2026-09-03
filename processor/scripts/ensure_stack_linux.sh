#!/usr/bin/env bash
# Keep the EgoData service chain converged after boot or a transient dependency
# failure.  This script never launches uvicorn or workers directly; it only
# asks the per-user systemd manager to start the canonical units.

set -u

STORAGE_ROOT="/home/stouching/.cache/egodata-data"
USERS_FILE="${STORAGE_ROOT}/state/users.json"
LOCK_FILE="/tmp/egodata-stack-coordinator.lock"

# A second copy must not be able to issue competing start requests.
exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

sleep_until_ready=5

storage_ready() {
    /usr/bin/mountpoint -q "${STORAGE_ROOT}" && /usr/bin/test -r "${USERS_FILE}"
}

database_tunnel_ready() {
    # Bash opens and closes the local forwarded port; no extra process or
    # package (nc/netcat) is needed.
    (echo > /dev/tcp/127.0.0.1/15432) >/dev/null 2>&1
}

backend_ready() {
    local response
    response=$(/usr/bin/curl --silent --show-error --fail \
        --connect-timeout 1 --max-time 3 \
        http://127.0.0.1:8000/health 2>/dev/null) || return 1
    [[ "${response}" == *'"status":"ok"'* || \
       "${response}" == *'"status": "ok"'* ]]
}

ensure_unit_started() {
    local unit="$1"
    local state
    state=$(systemctl --user show "${unit}" --property=ActiveState --value 2>/dev/null || true)

    # Do not enqueue another start while systemd is already activating or
    # stopping the unit.  systemd also coalesces identical jobs, but this
    # explicit guard keeps the retry loop quiet and deterministic.
    case "${state}" in
        active|activating|deactivating)
            return 0
            ;;
    esac

    systemctl --user --no-block start "${unit}" >/dev/null 2>&1 || true
}

while :; do
    if ! storage_ready || ! database_tunnel_ready; then
        sleep "${sleep_until_ready}"
        continue
    fi

    ensure_unit_started egodata-backend.service

    # Workers must only be started after the API has confirmed both the
    # PostgreSQL tunnel and the remote storage are usable.
    if backend_ready; then
        ensure_unit_started egodata-workers.service
    fi

    sleep "${sleep_until_ready}"
done
