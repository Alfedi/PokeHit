#!/bin/bash

wins=0

for _ in {0..96}; do
    if python ./test.py | grep -i MCTS; then
	wins=$((wins+1))
    fi
done

printf "De 96 ejecuciones, ha ganado %d" $wins
