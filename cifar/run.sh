#!/bin/bash

python server.py  &
sleep 5 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect
for ((nb=0; nb<$NBCLIENTS; nb++))
do
    python client_cifar.py &
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400