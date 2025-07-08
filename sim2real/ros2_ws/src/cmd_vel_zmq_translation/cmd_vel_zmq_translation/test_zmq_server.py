#!/usr/bin/env python3
import zmq, time, sys
ctx = zmq.Context()
s = ctx.socket(zmq.PULL); s.bind("tcp://127.0.0.1:6969")
while True:
    print("RECV:", s.recv_string())
