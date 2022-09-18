import socket
import string
 
s = socket.socket()
host = socket.gethostname()
port = 2333
s.connect((host, port))
buf = s.recv(32)
print(buf.decode())
s.send(b'getblockheight')
buf = s.recv(32)
print(buf.decode())
s.send(b'123')
buf = s.recv(32)
print(buf.decode())
s.close()