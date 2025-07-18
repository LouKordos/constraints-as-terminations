#!/bin/bash
set -x
echo "First, ensure that you ran the following on the remote ssh server that has internet and NTP access, use same ntp server in the name as defined for the workstation:"
echo "socat tcp4-listen:9123,reuseaddr,fork udp4:0.ubuntu.pool.ntp.org:123"

(sleep 10 && sudo socat udp4-recvfrom:123,bind=127.0.0.1,fork tcp:localhost:9123) & # Dirty way because ssh will block and we want to automatically start socat
echo "Restart chrony in a new terminal after the following command to ensure time is synced"
WORKSTATION_USER=kordoslo
WORKSTATION_IP=192.168.123.222
ssh -N -L 9123:localhost:9123 $WORKSTATION_USER@$WORKSTATION_IP

