#!/usr/bin/env bash

set +e

date --iso-8601=seconds
hostname
pwd
id
groups
umask
sudo -n -l 2>&1
df -hT / /home /data /scratch 2>&1
findmnt -T /data/chenyiteng -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T /scratch/chenyiteng -o TARGET,SOURCE,FSTYPE,OPTIONS
ls -ld /home/chenyiteng /data/chenyiteng /scratch/chenyiteng /data/shared 2>&1
getent group sudo docker lxd labdata 2>&1
systemctl is-active docker containerd 2>&1
ls -l /var/run/docker.sock 2>&1
ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,cmd --sort=-rss | head -n 30
date --iso-8601=seconds
exit 0
