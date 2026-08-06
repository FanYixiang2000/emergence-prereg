#!/bin/bash
# Network watcher: when a Pythia-6.9B weight host becomes reachable,
# run the step0 smoke download and stop. Logs to outputs/.
cd "$(dirname "$0")"
while true; do
    for base in "https://huggingface.co" "https://hf-mirror.com"; do
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 \
            "$base/EleutherAI/pythia-6.9b/resolve/step0/config.json")
        if [ "$code" = "200" ] || [ "$code" = "302" ]; then
            echo "$(date -Is) REACHABLE via $base (HTTP $code)"
            if [ "$base" = "https://hf-mirror.com" ]; then
                export HF_ENDPOINT="https://hf-mirror.com"
            fi
            echo "$(date -Is) starting step0 smoke download"
            python predownload_pythia.py --size 6.9b --steps 0 \
                > outputs/pythia69b_smoke.log 2>&1
            rc=$?
            echo "$(date -Is) smoke download exit=$rc"
            if [ $rc -eq 0 ]; then
                echo "NETWORK_RESTORED_SMOKE_OK"
                exit 0
            fi
        else
            echo "$(date -Is) $base unreachable (HTTP $code)"
        fi
    done
    sleep 600
done
