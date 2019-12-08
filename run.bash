#!/bin/bash

PREFIX="docker-compose run --rm app bash -l"
OPTIONS=${@:2}

case $1 in
    "run" )
        ${PREFIX} scripts/run_scripts.bash bowi/methods/run.py ${OPTIONS}
        ;;
    "test" )
        ${PREFIX} scripts/run_tests.bash ${OPTIONS}
        ;;
    "lint" )
        ${PREFIX} scripts/lint.bash
        ;;
    "bash" )
        ${PREFIX}
        ;;
    "python" )
        ${PREFIX} scripts/run_scripts.bash ${OPTIONS}
        ;;
    "stub" )
        ${PREFIX} scripts/make_stub.bash
        ;;
    "trec" )
        PREC_FILE=$2
        DATASET=(${PREC_FILE//\// })
        DATASET=${DATASET[1]}
        docker-compose -f compose/trec/docker-compose.yaml run --rm trec trec_eval -m recall -q results/${DATASET}/gt.qrel $PREC_FILE
        ;;
    * )
        echo "Invalid option ${1}" ;;
esac
