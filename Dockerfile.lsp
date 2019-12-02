FROM hirota/bowi

ARG NEW_USER
ENV HOME=/home/${NEW_USER}

# Modify user name. Don't create a new user so that UID wouldn't conflict
USER root
RUN usermod -l ${NEW_USER} jovyan
RUN mkdir ${HOME}
RUN echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
VOLUME ${HOME}
USER ${NEW_USER}
