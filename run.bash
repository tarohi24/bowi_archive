#!/bin/bash

case $1 in
    "test" )
        if [ "${#@}" -eq 1 ]
        then
            options="/workplace/docsim/tests"
        else
            options=${@:2}
        fi
        docker-compose run --workdir="/workplace" -e IS_TEST=1 --rm python pytest ${options}
        ;;
    "bash" )
        docker-compose run --rm python bash
        ;;
    "python" )
        docker-compose run --rm python python
        ;;
    * )
        echo "Invalid option ${1}" ;;
esac
