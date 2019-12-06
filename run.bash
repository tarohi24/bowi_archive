#!/bin/bash

PREFIX="docker-compose run --rm app bash -l"

case $1 in
    "test" )
        options=${@:2}
        ${PREFIX} scripts/run_tests.bash ${options}
        ;;
    "lint" )
        ${PREFIX} scripts/lint.bash
        ;;
    "bash" )
        ${PREFIX}
        ;;
    "python" )
        ${PREFIX} scripts/run_scripts.bash ${@:2}
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
