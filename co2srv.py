# Copyright (C) 2022 by The CO2 Project
# SuperHacker UEFI
# Path: co2srv.py

from cgi import test
from codecs import getreader
from random import randint
from site import removeduppaths
import socket
from tkinter import E
from co2 import CO2Core as CO2Core
from threading import Thread
import os
import time

def genRandPow(length):
    buf = []
    for _ in range(length):
        num = randint(0, 19)
        if num < 10:
            buf.append(str(num))
        else:
            buf.append(chr(ord('a') + num - 10))
    return ''.join(buf)
    
class CO2Srv:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.hostip = socket.gethostbyname(self.hostname)
        self.port = 2333
        self.core = CO2Core()
        self.nodelist = []
        self.blacknodes = []
        self.miner_need_restart = False

    def removeJobByBlock(self, block):
        for i in range(len(self.core.jobs)):
            if self.core.jobs[i] == block[64:1024]:
                self.core.jobs.pop(i)
                break
    
    def listener(self):
        while True:
            s = socket.socket()
            s.bind((self.hostname, self.port))
            s.listen()
            c, addr = s.accept()
            c.settimeout(3.0)
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
                        self.removejobByblock(block)
                        self.core.chain.append(block)
                        self.miner_need_restart = True
                        c.send(b'Verified!')
                    else:
                        c.send(b'inValid Block!')
                else:
                    c.send(b'Old block!')  
            elif buf.decode() == 'getbalance':
                c.send(b'ready')
                address = c.recv(192).decode()
                c.send(str(self.core.checkBalance(address)).encode())
            elif buf.decode() == 'verifytx':
                c.send(b'ready')
                transaction = c.recv(960).decode()
                if transaction not in self.core.jobs and self.core.verifyTransaction(transaction):
                    self.core.jobs.append(transaction)
                    c.send(b'Verified!')
                else:
                    c.send(b'Error transaction!')
            elif buf.decode() == 'getnodes':
                c.send(str(self.nodelist).encode())
            s.close()
            if addr[0] != self.hostip and addr[0] != '127.0.0.1' and addr[0] not in self.nodelist:
                remoteH = self.testGetRemoteHeight(addr[0])
                if remoteH != -1:
                    print('Node %s added!' %addr[0])
                    self.nodelist.append(addr[0])
    
    def getRemoteHeight(self, nodeid):
        s = socket.socket()
        s.settimeout(1)
        s.connect((self.nodelist[nodeid], 2333))
        if s.recv(32).decode() == 'CO2Srv':
            s.send(b'getblockheight')
            buf = s.recv(32)
            s.close()
            return int(buf.decode())
        else:
            s.close()
            return -1
    
    def testGetRemoteHeight(self, nodeip):
        s = socket.socket()
        s.settimeout(1)
        s.connect((nodeip, 2333))
        if s.recv(32).decode() == 'CO2Srv':
            s.send(b'getblockheight')
            buf = s.recv(32)
            s.close()
            return int(buf.decode())
        else:
            s.close()
            return -1
    
    def boardcastJobs(self):
        for nodeid in range(len(self.nodelist)):
            remoteH = self.getRemoteHeight(nodeid)
            if(remoteH == len(self.core.chain)):
                for i in range(len(self.core.jobs)):
                    s = socket.socket()
                    s.settimeout(3.0)
                    s.connect((self.nodelist[nodeid], 2333))
                    if(s.recv(32).decode() == 'CO2Srv'):
                        s.send(b'verifytx')
                        buf = s.recv(32)
                        if buf.decode() == 'ready':
                            s.send(self.core.jobs[i].encode())
                            buf = s.recv(32)
                            if buf.decode() == 'Verified!':
                                print('job %d sent to %s' %(i, self.nodelist[nodeid]))
                            else:
                                print('job %d sent to %s failed!' %(i, self.nodelist[nodeid]))
                        s.close()
                    else:
                        s.close()
                        print('Node %s error!' %self.nodelist[nodeid])
                        break
            elif(remoteH  == -1):
                self.blacknodes.append(self.nodelist[nodeid])
                print('blacklisted node %s' %self.nodelist[nodeid])
        for blackid in self.blacknodes:
            # Cleanup blacknodes
            self.nodelist.pop(blackid)
        self.blacknodes = []

    def boardcastBlock(self):
        for nodeid in range(len(self.nodelist)):
            remoteH = self.getRemoteHeight(nodeid)
            if(remoteH == len(self.core.chain) - 1):
                s = socket.socket()
                s.settimeout(5.0)
                s.connect((self.nodelist[nodeid], 2333))
                if s.recv(32).decode() == 'CO2Srv':
                    s.send(b'verifyblock')
                    buf = s.recv(32)
                    if buf.decode() == 'ready':
                        s.send(self.core.chain[-1].encode())
                        buf = s.recv(32)
                        if buf.decode() == 'Verified!':
                            print('block sent to %s' %self.nodelist[nodeid])
                        else:
                            print('Node %s rejected our block!' %self.nodelist[nodeid])
                else:
                    print('Node %s error' %self.nodelist[nodeid])
                s.close()
            elif(remoteH == -1):
                self.blacknodes.append(nodeid)
                print('Node %s is blacklisted!' %self.nodelist[nodeid])
        for blackid in self.blacknodes:
            # Cleanup blacknodes
            self.nodelist.pop(blackid)
        self.blacknodes = []

    def syncHeight(self):
        for nodeid in range(len(self.nodelist)):
            remoteHeight = s.recv(32).decode()
            print('Remote height: %s' %remoteHeight)
            if int(remoteHeight) > len(self.core.chain):
                for i in range(len(self.core.chain), int(remoteHeight)):
                    s = socket.socket()
                    s.settimeout(3.0)
                    s.connect((self.nodelist[nodeid], 2333))
                    if(s.recv(32).decode() == 'CO2Srv'):
                        s.send(b'syncblock')
                        buf = s.recv(32)
                        if buf.decode() == 'n?':
                            s.send(str(i).encode())
                            buf = s.recv(1600).decode()
                            s.close()
                            if buf != 'Error id!':
                                if(self.core.verifyBlock(buf)):
                                    self.removeJobByBlock(buf)
                                    self.core.chain.append(buf)
                                    self.miner_need_restart = True
                                    print('Block %d synced from %s' %(i, self.nodelist[nodeid]))
                                else:
                                    print('Block %d verification failed!' %i)
                                    break
                            else:
                                print('Remote Error: Block %d not found!' %i)
                                break
                        else:
                            s.close()
                            print('Remote Node Error: %s' %buf.decode())
                            break
                    else:
                        s.close()
                        print('Remote Node %s Error' %self.nodelist[nodeid])
                        break
            elif remoteHeight == -1:
                self.blacknodes.append(nodeid)
                print('Node %s is blacklisted!' %self.nodelist[nodeid])
        for blackid in self.blacknodes:
            # Cleanup blacknodes
            self.nodelist.pop(blackid)

    def updateNodes(self):
        for node in self.nodelist:
            s = socket.socket()
            s.settimeout(3.0)
            s.connect((node, 2333))
            if(s.recv(32).decode() == 'CO2Srv'):
                s.send(b'getnodes')
                remoteNodes = eval(s.recv(1024).decode())
                s.close()
                for rnode in remoteNodes:
                    if rnode != self.hostip and rnode != '127.0.0.1' and rnode not in self.nodelist:
                        self.nodelist.append(rnode)
                        print('Node %s added!' %rnode)

    def Miner(self, myAddr):
        while True:
            if len(self.core.jobs) != 0:
                myjob = self.core.jobs[-1]
                myPoWPayload = genRandPow(512)
                myBlock = self.core.makeBlock(myjob, myAddr, myPoWPayload)
                myBlockHash = myBlock[1744:1808]
                while not (self.miner_need_restart or self.core.verifyPoW(myPoWPayload, myBlockHash)):
                    myPoWPayload = genRandPow(512)
                    myBlock = self.core.makeBlock(myjob, myAddr, myPoWPayload)
                    myBlockHash = myBlock[1744:1808]
                if self.miner_need_restart: # Shit No goto command in python caused this!
                    self.miner_need_restart = False
                    continue
                if self.core.verifyBlock(myBlock):
                    self.core.chain.append(myBlock) # make main thread to publish!
                    self.removeJobByBlock(myBlock)
                    print('Block %d mined!' %len(self.core.chain))
            else:
                whiteTrans = self.core.makeWhiteTransaction()
                myPoWPayload = genRandPow(512)
                myBlock = self.core.makeBlock(whiteTrans, myAddr, myPoWPayload)
                myBlockHash = myBlock[1744:1808]
                while not (self.miner_need_restart or self.core.verifyPoW(myPoWPayload, myBlockHash)):
                    myPoWPayload = genRandPow(512)
                    myBlock = self.core.makeBlock(whiteTrans, myAddr, myPoWPayload)
                    myBlockHash = myBlock[1744:1808]
                if self.miner_need_restart: # Shit No goto command in python caused this!
                    self.miner_need_restart = False
                    continue
                if self.core.verifyBlock(myBlock):
                    self.core.chain.append(myBlock)
                    print('White Block %d mined!' %len(self.core.chain))
                else:
                    print('White Block verification failed!')
            time.sleep(5)

    def optimDiff(self):
        if(len(self.core.chain) > 1):
            lastBlock = self.core.chain[-1]
            lastBlockTime = int(lastBlock[64:80])
            nowTime = time.time()
            nowTimeInt = int(nowTime)
            if(nowTimeInt - lastBlockTime > self.core.blockTime): # Too slow
                if((nowTimeInt - lastBlockTime) % self.core.blockTimeOptimVarA == 0): # is timeout level to optim
                    #if(nowTime - nowTimeInt < 0.5): # is time to optim
                        self.core.diff = self.core.diff * (1 - self.core.blockTimeOptimVarRate)
                        print('Difficulty decreased to %f' %self.core.diff)
                        return
        if(len(self.core.chain) > 2):
            lastBlock = self.core.chain[-1]
            lastBlockTime = int(lastBlock[64:80])
            lastBlock2 = self.core.chain[-2]
            lastBlock2Time = int(lastBlock2[64:80])
            nowTime = time.time()
            nowTimeInt = int(nowTime)
            if(lastBlockTime - lastBlock2Time < self.core.blockTime): # Too fast
                if(nowTimeInt - lastBlockTime < self.core.blockTime): # considering optim in blockTime
                    if((nowTimeInt - lastBlockTime) % self.core.blockTimeOptimVarB == 0):
                        self.core.diff = self.core.diff * (1 + self.core.blockTimeOptimVarRate)
                        print('Difficulty increased to %f' %self.core.diff)
                        return



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
        mySrv.optimDiff()
        if len(mySrv.core.chain) > currentHeight:
            currentHeight = len(mySrv.core.chain)
            mySrv.boardcastBlock()
        if len(mySrv.core.jobs) > currentJobs:
            currentJobs = len(mySrv.core.jobs)
            mySrv.boardcastJobs()
        if len(mySrv.core.jobs) < currentJobs:
            currentJobs = len(mySrv.core.jobs)
        run_epoch += 1
        run_epoch %= 100
        if run_epoch % 10 == 0:
            mySrv.updateNodes()
            mySrv.syncHeight()
            for i in mySrv.nodelist:
                open('nodelist.txt', 'a').write(i + '\n')
            for i in range(fileHeight, len(mySrv.core.chain)):
                open('chain.txt', 'a').write(mySrv.core.chain[i] + '\n')
            fileHeight = len(mySrv.core.chain)
        time.sleep(1)
    
            

  