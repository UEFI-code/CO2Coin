# Copyright (C) 2022 by The CO2 Project
# SuperHacker UEFI
# Path: co2client.py

import socket
import string
import co2
import os
import time
class CO2Client:
    def __init__(self):
        self.core = co2.CO2Core(loadDL=False)
        self.node = socket.gethostname()
        self.port = 2333
    
    def getBlockHeight(self):
        s = socket.socket()
        s.settimeout(3)
        s.connect((self.node, self.port))
        if s.recv(32).decode() == 'CO2Srv':
            s.send(b'getblockheight')
            buf = s.recv(32)
            s.close()
            return int(buf.decode())
        else:
            print('node Error!')

    def publishBlock(self, block):
        s = socket.socket()
        s.connect((self.node, self.port))
        if s.recv(32).decode() == 'CO2Srv':
            s.send(b'verifyblock')
            buf = s.recv(32)
            if buf.decode() == 'ready':
                s.send(block.encode())
                buf = s.recv(32)
                if buf.decode() == 'Verified!':
                    print('block sent to %s' %self.node)
                else:
                    print('block verification failed!')
            else:
                print('node not ready!')
        else:
            print('node Error!')
        s.close()
    
    def publishTransaction(self, tx):
        s = socket.socket()
        s.connect((self.node, self.port))
        if s.recv(32).decode() == 'CO2Srv':
            s.send(b'verifytx')
            buf = s.recv(32)
            if buf.decode() == 'ready':
                s.send(tx.encode())
                buf = s.recv(32)
                if buf.decode() == 'Verified!':
                    print('tx sent to %s' %self.node)
                else:
                    print('tx verification failed!')
            else:
                print('node not ready!')
        else:
            print('node Error!')
        s.close()
    
    def getBalance(self, address):
        s = socket.socket()
        s.settimeout(3)
        s.connect((self.node, self.port))
        if s.recv(32).decode() == 'CO2Srv':
            s.send(b'getbalance')
            buf = s.recv(32)
            if buf.decode() == 'ready':
                s.send(address.encode())
                buf = s.recv(32)
                s.close()
                return int(buf.decode())
            else:
                print('node not ready!')
        else:
            print('node Error!')
        s.close()
        return False

if __name__ == '__main__':
    print('EarthCooler CO2Client v0.1')
    my_key = 'd10e1416590b938fc8e57088071af877d81ff20dc39707caa3866894a2a80eb59f249184a6e7f2ac25148082aeadbb24'
    my_addr = '6d731960af08e06ba3e84f660af4af9ccde5d47ba9070602a27687ba6e39b5cb778adcbca6fdef19e99edef115ec6fc9711867ae7e185922596c246f0734640d857c2d7e3f4d87aca6b02f7a891bdccd50ab712af753f0c7d987534bd94fab2b'
    target_addr = '75d6c47c5287e9b81fb8472d9ebf6635b5e20f5156d4bacb395997767790120b21fbd9ca67d7eed13ff68e3d8de25bd41ffa661742dcd41a00d83f02bcffd0205f248ea2dcb9432454de53097c78f79e6e0f50f5831ddb8ef4e57e7d1288e82e'
    myClient = CO2Client()

    myBalance = myClient.getBalance(my_addr)
    targetBalance = myClient.getBalance(target_addr)

    print('Our balance before tx: %d' %myBalance)
    print('Target balance before tx: %d' %targetBalance)

    if myBalance < 234:
        print('!!!We have Not enough balance!!!')
    
    blockID = str(myClient.getBlockHeight()).zfill(16)
    print('BlockID: %s' %blockID)
    tx = myClient.core.makeTransaction(blockID, my_addr, target_addr, 233, 'n' * 320, my_key)
    print(tx)
    myClient.publishTransaction(tx)
    time.sleep(30)
    print('Balance after tx: %d' %myClient.getBalance(target_addr))

    os.system('pause')
    