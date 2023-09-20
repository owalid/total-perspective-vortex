import yaml
from socket import socket
import os
from yaml.loader import SafeLoader
import logging
from sys import stdout
from colorama import *

def receive_message(conn: socket, buffer_size=1024):
    i = 0
    r = b''
    while True:
        try:
            data = conn.recv(buffer_size)
            r += data
            if len(data.decode()) < buffer_size or data == b'':
                return None if not r else r
            i += 1
            if i > 3:
                conn.sendall(b"This action is not allowed\n")
                return None
        except:
            return None if not r else r
    return None if not r else r

def send_error(conn, message, end=False):
    conn.sendall(f"\n{Back.RED}{Style.BRIGHT}Error:{Style.RESET_ALL} {message} Please contact your administrator.\n\n".encode())
    if end:
        conn.sendall(b'> ')

def send_message(conn, message, end=False):
    if type(message) == str:
        message = message.encode()
    conn.sendall(message)
    if end:
        conn.sendall(b'> ')