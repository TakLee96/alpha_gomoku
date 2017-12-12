#!/bin/bash
LST=(0 21 40 60 84 107 126 140)

for a in "${LST[@]}"; do
    for b in "${LST[@]}"; do
        if [[ $a -lt $b ]]; then
            python3 pgn.py dagger $a dagger $b >> all.pgn
        fi
    done
    python3 pgn.py dagger $a supervised 4000 >> all.pgn
done
