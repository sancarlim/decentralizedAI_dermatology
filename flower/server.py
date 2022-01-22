#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server.py
# Modified   : 22.01.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

import sys
sys.path.append('/workspace/flower')
import flwr as fl 

if __name__ == "__main__":
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=fl.server.strategy.FedAvg())