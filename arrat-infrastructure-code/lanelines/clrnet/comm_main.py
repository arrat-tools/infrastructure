import subprocess, shlex
import socket
import time
import struct

MY_IP = "172.17.0.2"
OTHER_IP = "172.17.0.1"
MY_PORT = 5005
TO_PORT = 5006

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((MY_IP, MY_PORT))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    if addr[0] == OTHER_IP:
    	command_id, = struct.unpack("<H", data[0:2])   
    	command = data[2:].decode()   
    	print("=============================")
    	print("Received command from host: %s" % command_id, command)
    	#cmd_str = "ls" + " " + command
    	print("Calling subprocess to execute:", command, "\n")
    	p = subprocess.call(shlex.split(command))
    	sock.sendto(struct.pack("<H", command_id), (OTHER_IP, TO_PORT))
    	print("Sent ID back to host:", command_id)
    	print("")
    	print("Subprocess finished")
    	print("=============================") 

