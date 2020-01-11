eval "$(pyenv init -)"
jupyter notebook --no-browser --NotebookApp.token=${JNOTE_TOKEN} --ip=0.0.0.0
