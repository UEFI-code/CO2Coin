from random import randint
from site import removeduppaths
import socket
from tkinter import E
from co2 import CO2Core as CO2Core
from threading import Thread
import os
import time

class CO2Srv:
    def __init__(self):
        self.host = socket.gethostname()
        self.port = 2333
        self.core = CO2Core()
        self.nodelist = []
        self.miner_need_restart = False

    def listener(self):
        while True:
            s = socket.socket()
            s.bind((self.host, self.port))
            s.listen()
            c, addr = s.accept()
            c.settimeout(3.0)
            if addr[0] != self.host and addr[0] not in self.nodelist:
                self.nodelist.append(addr[0])
            c.send(b'CO2Srv')
            buf = c.recv(32)
            if buf.decode() == 'getblockheight':
                c.send(str(len(self.core.chain)).encode())
            elif buf.decode() == 'syncblock':
                c.send(b'n?')
                id = int(c.recv(32).decode())
                if id < len(self.core.chain):
                    c.send(self.core.chain[id].encode())
                else:
                    c.send(b'Error id!')
            elif buf.decode() == 'verifyblock':
                c.send(b'ready')
                block = c.recv(1600).decode()
                if block not in self.core.chain:
                    if self.core.verifyBlock(block):
                        self.core.chain.append(block)
                        for i in range(len(self.core.jobs)):
                            if self.core.jobs[i] == block[64:1024]:
                                self.core.jobs.pop(i)
                                break
                        c.send(b'Verified!')
                    else:
                        c.send(b'Verification failed!')
                else:
                    c.send(b'Old block!')  
            elif buf.decode() == 'getbalance':
                c.send(b'ready')
                address = c.recv(192).decode()
                c.send(str(self.core.checkBalance(address)).encode())
            elif buf.decode() == 'verifytransaction':
                c.send(b'ready')
                transaction = c.recv(960).decode
                if transaction not in self.core.jobs and self.core.verifyTransaction(transaction):
                    self.core.jobs.append(transaction)
                    c.send(b'Verified!')
                else:
                    c.send(b'Error transaction!')
            elif buf.decode() == 'getnodes':
                c.send(str(self.nodelist).encode())
            s.close()    

    def boardcastJobs(self):
        for node in self.core.nodelist:
            s = socket.socket()
            for i in range(len(self.core.jobs)):
                s.connect((node, 2333))
                s.send(b'verifytransaction')
                buf = s.recv(32)
                if buf.decode() == 'ready':
                    s.send(self.core.jobs[i].encode())
                    buf = s.recv(32)
                    if buf.decode() == 'Verified!':
                        print('job %d sent to %s' %(i, node))
            s.close()

    def boardcastBlock(self):
        for node in self.core.nodelist:
            s = socket.socket()
            s.settimeout(1.0)
            s.connect((node, 2333))
            s.send(b'verifyblock')
            buf = s.recv(32)
            if buf.decode() == 'ready':
                s.send(self.core.chain[-1].encode())
                buf = s.recv(32)
                if buf.decode() == 'Verified!':
                    print('block sent to %s' %node)
            s.close()

    def syncHeight(self):
        for node in self.nodelist:
            s = socket.socket()
            s.settimeout(3.0)
            s.connect((node, 2333))
            if(s.recv(32).decode() == 'CO2Srv'):
                s.send(b'getblockheight')
                s.close()
                remoteHeight = s.recv(32).decode()
                print('Remote height: %s' %remoteHeight)
                if int(remoteHeight) > len(self.core.chain):
                    for i in range(len(self.core.chain), int(remoteHeight)):
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
                                if(self.core.verifyBlock(buf)):
                                    self.core.chain.append(buf)
                                    print('Block %d synced from %s' %(i, node))
                                else:
                                    print('Block %d verification failed!' %i)
                                    break
                            else:
                                print('Remote Error: Block %d not found!' %i)
                                break

    def updateNodes(self):
        host = socket.gethostname()
        for node in self.nodelist:
            s = socket.socket()
            s.settimeout(3.0)
            s.connect((node, 2333))
            if(s.recv(32).decode() == 'CO2Srv'):
                s.send(b'getnodes')
                remoteNodes = eval(s.recv(1024).decode())
                s.close()
                for rnode in remoteNodes:
                    if rnode != self.host and rnode not in self.nodelist:
                        self.nodelist.append(rnode)
                        print('Node %s added!' %rnode)

    def Miner(self, myAddr):
        while True:
            if len(self.core.jobs) != 0:
                myjob = self.core.jobs[-1]
                myPoW = str(randint(0, 2**256)).zfill(512)
                while not (self.miner_need_restart or self.core.verifyPoW(myPoW)):
                    myPoW = str(randint(0, 2**256)).zfill(512)
                if self.miner_need_restart:
                    self.miner_need_restart = False
                    continue
                myblock = self.core.makeBlock(myjob, myAddr, myPoW)
                if self.core.verifyBlock(myblock):
                    print('Block %d mined!' %len(self.core.chain))
                    # Search for old job to delete
                    for i in range(len(self.core.jobs)):
                        if self.core.jobs[i] == myjob:
                            self.core.jobs.pop(i)
                            break
                    self.boardcastBlock()
                    self.core.chain.append(myblock)
            else:
                myPoW = str(randint(0, 2**256)).zfill(512)
                while not (self.miner_need_restart or self.core.verifyPoW(myPoW)):
                    myPoW = str(randint(0, 2**256)).zfill(512)
                if self.miner_need_restart:
                    self.miner_need_restart = False
                    continue
                mined = self.core.makeWhiteMiningReward(myAddr, myPoW)
                if mined:
                    print('White mined!')
                else:
                    print('White mining failed!')
                time.sleep(1)

if __name__ == '__main__':
    print('CO2Srv started.')
    myAddr = '6d731960af08e06ba3e84f660af4af9ccde5d47ba9070602a27687ba6e39b5cb778adcbca6fdef19e99edef115ec6fc9711867ae7e185922596c246f0734640d857c2d7e3f4d87aca6b02f7a891bdccd50ab712af753f0c7d987534bd94fab2b'
    fileHeight = 0
    mySrv = CO2Srv()
    if(os.path.exists('chain.txt')):
        mySrv.core.chain = open('chain.txt', 'r').readlines()
        fileHeight = len(mySrv.core.chain)
    if(os.path.exists('nodelist.txt')):
        mySrv.nodelist = open('nodelist.txt', 'r').readlines()
    mySrv.updateNodes()
    mySrv.syncHeight()
    currentHeight = len(mySrv.core.chain)
    currentJobs = len(mySrv.core.jobs)
    net = Thread(target=mySrv.listener)
    net.start()
    miner = Thread(target=mySrv.Miner, args=(myAddr,))
    miner.start()
    run_epoch = 0
    while True:
        if len(mySrv.core.chain) > currentHeight:
            mySrv.miner_need_restart = True
            currentHeight = len(mySrv.core.chain)
        if len(mySrv.core.jobs) > currentJobs:
            mySrv.boardcastJobs()
        run_epoch += 1
        run_epoch %= 100
        if run_epoch % 10 == 0:
            for i in mySrv.nodelist:
                open('nodelist.txt', 'a').write(i + '\n')
            for i in range(fileHeight, len(mySrv.core.chain)):
                open('chain.txt', 'a').write(mySrv.core.chain[i] + '\n')
            fileHeight = len(mySrv.core.chain)
        time.sleep(1)
    
            

  