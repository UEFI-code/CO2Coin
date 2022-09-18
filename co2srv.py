from random import random
from site import removeduppaths
import socket
from co2 import CO2Core as CO2Core
from threading import Thread
import os
import time

nodelist = []
co2 = CO2Core()

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
        s.settimeout(1.0)
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

def Miner(myAddr):
    while True:
        if len(co2.jobs) != 0:
            myjob = co2.jobs[-1]
            myPoW = str(random.randint(0, 2**256)).zfill(512)
            while not co2.verifyPoW(myPoW):
                myPoW = str(random.randint(0, 2**256)).zfill(512)
            myblock = co2.makeBlock(myjob, myAddr, myPoW)
            if co2.verifyBlock(myblock):
                print('Block %d mined!' %len(co2.chain))
                for i in range(len(co2.jobs)):
                    if co2.jobs[i] == myjob:
                        co2.jobs.pop(i)
                        break
                boardcastBlock()
                co2.chain.append(myblock)
        else:
            
            mined = co2.makeWhiteMiningReward(myAddr, '0' * 512)
            if mined:
                print('White mined!')
                print(co2.chain[-1])
            else:
                print('White mining failed!')
            time.sleep(1)

if __name__ == '__main__':
    print('CO2Srv started.')
    myAddr = '6d731960af08e06ba3e84f660af4af9ccde5d47ba9070602a27687ba6e39b5cb778adcbca6fdef19e99edef115ec6fc9711867ae7e185922596c246f0734640d857c2d7e3f4d87aca6b02f7a891bdccd50ab712af753f0c7d987534bd94fab2b'
    fileHeight = 0
    if(os.path.exists('chain.txt')):
        co2.chain = open('chain.txt', 'r').readlines()
        fileHeight = len(co2.chain)
    if(os.path.exists('nodelist.txt')):
        nodelist = open('nodelist.txt', 'r').readlines()
    updateNodes()
    syncHeight()
    currentHeight = len(co2.chain)
    currentJobs = len(co2.jobs)
    net = Thread(target=listener)
    net.start()
    miner = Thread(target=Miner, args=(myAddr,))
    miner.start()
    run_epoch = 0
    while True:
        if len(co2.chain) > currentHeight:
            #miner.stop()
            currentHeight = len(co2.chain)
            #miner.start()
        if len(co2.jobs) > currentJobs:
            boardcastJobs()
        run_epoch += 1
        run_epoch %= 100
        if run_epoch % 10 == 0:
            for i in nodelist:
                open('nodelist.txt', 'a').write(i + '\n')
            for i in range(fileHeight, len(co2.chain)):
                open('chain.txt', 'a').write(co2.chain[i] + '\n')
            fileHeight = len(co2.chain)
        time.sleep(1)
    
            

  