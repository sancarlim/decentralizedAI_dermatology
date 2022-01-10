#!/bin/bash

# Loading script arguments
NBCLIENTS="${1:-2}" # Nb of clients launched by the script (default to 2)
NBMINCLIENTS="${2:-2}" # Nb min of clients before launching round (default to 2)
NBFITCLIENTS="${3:-2}" # Nb of clients sampled for the round (default to 2)
NBROUNDS="${4:-3}" # Nb of rounds (default to 3)

python server.py -r $NBROUNDS -fc $NBFITCLIENTS -ac $NBMINCLIENTS &
sleep 5 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect
for ((nb=0; nb<$NBCLIENTS; nb++))
do
    python client_cifar.py &
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400