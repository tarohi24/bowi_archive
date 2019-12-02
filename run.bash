#!/bin/bash

case $1 in
    "test" )
        options=${@:2}
        docker-compose run --rm python bash scripts/run_tests.bash ${options}
        ;;
    "bash" )
        docker-compose run --rm python bash -l
        ;;
    "python" )
        docker-compose run --rm python bash scripts/run_scripts.bash ${@:2}
        ;;
    * )
        echo "Invalid option ${1}" ;;
esac
