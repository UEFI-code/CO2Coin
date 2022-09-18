import socket
import string
 
s = socket.socket()
host = socket.gethostname()
port = 2333
s.connect((host, port))
buf = s.recv(32).decode()
print(buf.split(',')[0])


s.send(b'getnodes')
buf = s.recv(128).decode()
print(buf.split()[0][2:-2])
print(eval(buf)[0])
s.send(b'75d6c47c5287e9b81fb8472d9ebf6635b5e20f5156d4bacb395997767790120b21fbd9ca67d7eed13ff68e3d8de25bd41ffa661742dcd41a00d83f02bcffd0205f248ea2dcb9432454de53097c78f79e6e0f50f5831ddb8ef4e57e7d1288e82e')
buf = s.recv(32)
print(buf.decode())
s.close()
