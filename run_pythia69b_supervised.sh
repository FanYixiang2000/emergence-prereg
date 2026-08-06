#!/bin/bash
# Supervisor for the 6.9B S8 run: waits for the proxy tunnel, then
# (re)starts the collapse probe; the downloader resumes from its range
# journal. When the collapse summary exists, runs the tail probe, then
# the checkpoint hash audit. Idempotent; safe to restart.
cd "$(dirname "$0")"
export http_proxy=http://127.0.0.1:18060
export https_proxy=http://127.0.0.1:18060
export PYTHIA_BASE_URL=https://hf-mirror.com
export PYTHIA_DOWNLOAD_THREADS=24
export PYTHIA_RANGE_TIMEOUT=180
export PYTHIA_CKPT_ATTEMPTS=10

proxy_ok() {
    code=$(curl -s -x http://127.0.0.1:18060 -o /dev/null \
        -w "%{http_code}" --max-time 20 \
        "https://hf-mirror.com/EleutherAI/pythia-6.9b/resolve/step0/config.json")
    [ "$code" = "200" ] || [ "$code" = "302" ] || [ "$code" = "307" ]
}

while true; do
    if [ ! -f outputs/pythia_collapse_summary_6.9b.json ]; then
        if proxy_ok; then
            echo "$(date -Is) proxy OK; starting collapse probe"
            python pythia_collapse_probe.py --size 6.9b \
                --device cuda:0 --dtype bfloat16 --keep_checkpoints \
                >> outputs/pythia69b_collapse2.log 2>&1
            echo "$(date -Is) collapse probe exited ($?)"
        else
            echo "$(date -Is) proxy down; waiting"
        fi
        sleep 120
        continue
    fi
    if [ ! -f outputs/pythia_tail_summary_6.9b.json ]; then
        echo "$(date -Is) collapse done; starting tail probe"
        python pythia_tail_gradualism.py --size 6.9b \
            --device cuda:0 --dtype bfloat16 \
            >> outputs/pythia69b_tail.log 2>&1
        echo "$(date -Is) tail probe exited ($?)"
        sleep 30
        continue
    fi
    echo "$(date -Is) both probes done; running hash audit"
    python verify_pythia_checkpoints.py --size 6.9b \
        >> outputs/pythia69b_hash_audit.log 2>&1
    echo "PYTHIA69B_PIPELINE_DONE"
    break
done
