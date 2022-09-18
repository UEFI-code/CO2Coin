from site import removeduppaths
import socket
import co2
from threading import Thread
import os
nodelist = []

def listener():
    host = socket.gethostname()
    port = 2333
    while True:
        s = socket.socket()
        s.bind((host, port))
        s.listen()
        c, addr = s.accept()
        c.settimeout(3.0)
        if addr[0] != host and addr[0] not in nodelist:
            nodelist.append(addr[0])
        c.send(b'CO2Srv')
        buf = c.recv(32)
        if buf.decode() == 'getblockheight':
            c.send(str(len(co2.chain)).encode())
        elif buf.decode() == 'syncblock':
            c.send(b'n?')
            id = int(c.recv(32).decode())
            if id < len(co2.chain):
                c.send(co2.chain[id].encode())
            else:
                c.send(b'Error id!')
        elif buf.decode() == 'verifyblock':
            c.send(b'ready')
            block = c.recv(1600).decode()
            if block not in co2.chain:
                if co2.verifyBlock(block):
                    co2.chain.append(block)
                    for i in range(len(co2.jobs)):
                        if co2.jobs[i] == block[64:1024]:
                            co2.jobs.pop(i)
                            break
                    c.send(b'Verified!')
                else:
                    c.send(b'Verification failed!')
            else:
                c.send(b'Old block!')  
        elif buf.decode() == 'getbalance':
            c.send(b'ready')
            address = c.recv(192).decode()
            c.send(str(co2.checkBalance(address)).encode())
        elif buf.decode() == 'verifytransaction':
            c.send(b'ready')
            transaction = c.recv(960).decode
            if transaction not in co2.jobs and co2.verifyTransaction(transaction):
                co2.jobs.append(transaction)
                c.send(b'Verified!')
            else:
                c.send(b'Error transaction!')
        elif buf.decode() == 'getnodes':
            c.send(str(nodelist).encode())
        s.close()    

def boardcastJobs():
    for node in nodelist:
        s = socket.socket()
        for i in range(len(co2.jobs)):
            s.connect((node, 2333))
            s.send(b'verifytransaction')
            buf = s.recv(32)
            if buf.decode() == 'ready':
                s.send(co2.jobs[i].encode())
                buf = s.recv(32)
                if buf.decode() == 'Verified!':
                    print('job %d sent to %s' %(i, node))
        s.close()

def boardcastBlock():
    for node in nodelist:
        s = socket.socket()
        s.connect((node, 2333))
        s.send(b'verifyblock')
        buf = s.recv(32)
        if buf.decode() == 'ready':
            s.send(co2.chain[-1].encode())
            buf = s.recv(32)
            if buf.decode() == 'Verified!':
                print('block sent to %s' %node)
        s.close()

def syncHeight():
    for node in nodelist:
        s = socket.socket()
        s.settimeout(3.0)
        s.connect((node, 2333))
        if(s.recv(32).decode() == 'CO2Srv'):
            s.send(b'getblockheight')
            s.close()
            remoteHeight = s.recv(32).decode()
            print('Remote height: %s' %remoteHeight)
            if int(remoteHeight) > len(co2.chain):
                for i in range(len(co2.chain), int(remoteHeight)):
                    s = socket.socket()
                    s.settimeout(3.0)
                    s.connect((node, 2333))
                    s.send(b'syncblock')
                    buf = s.recv(32)
                    if buf.decode() == 'n?':
                        s.send(str(i).encode())
                        buf = s.recv(1600).decode()
                        s.close()
                        if buf != 'Error id!':
                            if(co2.verifyBlock(buf)):
                                co2.chain.append(buf)
                                print('Block %d synced from %s' %(i, node))
                            else:
                                print('Block %d verification failed!' %i)
                                break
                        else:
                            print('Remote Error: Block %d not found!' %i)
                            break

def updateNodes():
    host = socket.gethostname()
    for node in nodelist:
        s = socket.socket()
        s.settimeout(3.0)
        s.connect((node, 2333))
        if(s.recv(32).decode() == 'CO2Srv'):
            s.send(b'getnodes')
            remoteNodes = eval(s.recv(1024).decode())
            s.close()
            for node in remoteNodes:
                if node != host and node not in nodelist:
                    nodelist.append(node)
                    print('Node %s added!' %node)

if __name__ == '__main__':
    print('CO2Srv started.')
    if(os.path.exists('chain.txt')):
        co2.chain = open('chain.txt', 'r').readlines()
    if(os.path.exists('nodelist.txt')):
        nodelist = open('nodelist.txt', 'r').readlines()
    updateNodes()
    syncHeight()
    net = Thread(target=listener)
    net.start()
    while True:
        if len(co2.jobs) != 0:
            myjob = co2.jobs[-1]
            co2.makeBlock(myjob)

  