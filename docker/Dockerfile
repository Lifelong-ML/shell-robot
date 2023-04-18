FROM ros:noetic
ENV HISTFILESIZE=
ENV HISTSIZE=
ENV HISTTIMEFORMAT="[%F %T] "
# Change the file location because certain bash sessions truncate .bash_history file upon close.
# http://superuser.com/questions/575479/bash-history-truncated-to-500-lines-on-each-login
ENV HISTFILE=/root/.bash_history

RUN apt update -y
RUN apt install python3-pip python3-matplotlib python3-pillow -y
RUN apt install python3-opencv -y

WORKDIR /project