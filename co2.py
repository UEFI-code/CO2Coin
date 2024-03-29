# Copyright (C) 2022 by The CO2 Project
# SuperHacker UEFI
# Path: co2.py

from time import time
from ecdsa import SigningKey, VerifyingKey, NIST384p
import hashlib
import proteinFold
import MBEPredictor

def sha256(msg):
    return hashlib.sha256(msg.encode()).hexdigest()

def generate_keys():
    sk = SigningKey.generate(curve=NIST384p)
    vk = sk.get_verifying_key()
    return sk.to_string().hex(), vk.to_string().hex()

def sign(msg, sk):
    return SigningKey.from_string(bytes.fromhex(sk), curve=NIST384p).sign(msg.encode()).hex()

def verifySig(msg, sig, vk):
    try:
        return VerifyingKey.from_string(bytes.fromhex(vk), curve=NIST384p).verify(bytes.fromhex(sig), msg.encode())
    except:
        return False

class CO2Core:
    def __init__(self, loadDL = True):
        self.chain = []
        self.jobs = []
        self.fee = 1
        self.rewardcoins = 128
        self.diff = 0.01
        self.blockTime =  30
        self.lastTimeCalcBlock = 0
        if loadDL:
            self.proteinFolder = proteinFold.ProteinFold3D(20, 64, 0)
            self.MBEPredictor = MBEPredictor.MBEPredictor()
            self.CO2Model = MBEPredictor.loadCO2Model()

    def checkBalance(self, address):
        balance = 0
        for block in self.chain:
            blockData = block[96:1056] # 960 bytes
            sender = blockData[0:192]
            receiver = blockData[192:384]
            amount = blockData[384:448]
            miner = block[1056:1248]
            if sender == address:
                balance -= (int(amount) + self.fee)
            if receiver == address:
                balance += int(amount)
            if miner == address:
                balance += self.rewardcoins
        return balance

    # def checkDoubleSpend(self, transaction):
    #     for block in self.chain:
    #         blockData = block[80:1040]
    #         if blockData == transaction:
    #             return True
    #     return False

    def verifyTransaction(self, transaction):
        # if self.checkDoubleSpend(transaction):
        #     return False
        blockID = str(len(self.chain)).zfill(16)
        sender = transaction[0:192]
        receiver = transaction[192:384]
        if sender == "0" * 192 and receiver == "0" * 192:
            if(len(self.jobs) != 0):
                print('Pls verify blocks, don\'t do white mining!')
                return False
            print("No jobs to verify, but may give you white mined reward!")
            print('White Transaction Public Key is Missing, bypass this check!')
            return True
        
        amount = transaction[384:448] # 64 bytes
        asset = transaction[448:768]  # 320 bytes
        if(int(amount) <= 0 or (self.checkBalance(sender) - self.fee) < int(amount)):
            #return False
            resultMoney = False
        else:
            resultMoney= True
        sig = transaction[768:960]  # 192 bytes
        hash = sha256(blockID + sender + receiver + amount + asset)
        resultSig = verifySig(hash, sig, sender)

        print('----verfiyTransaction Debug----')
        print('Current Work blockID: %s' %blockID)
        print('Sender: %s' %sender)
        print('Receiver: %s' %receiver)
        print('Amount: %s' %amount)
        print('Balance: %d' %self.checkBalance(sender))
        print('IsMoneyValid: %s' %resultMoney)
        print('Asset: %s' %asset)
        print('Sig: %s' %sig)
        print('Hash: %s' %hash)
        print('IsSignatureValid: %s' %resultSig)
        print('----End of Debug----')

        return resultSig and resultMoney

    def verifyPoW(self, payload, hash):
        if len(payload) != 512:
            return False, 0
        if len(hash) != 64:
            print("Hash length is wrong!" + str(len(hash)))
            return False, 0
        # for i in self.chain:
        #     if i[1216:1792] == payload:
        #         print("Not your PoW!")
        #         return False, 0
        #RNA = []
        PepChain = []
        try:    
            for i in range(512):
                # numA = int(payload[i], 16) % 4
                # numB = (int(payload[i], 16) >> 2) % 4
                # RNA.append(numA)
                # RNA.append(numB)
                num = int(payload[i], 20)
                PepChain.append(num)
                if i % 16 == 0:
                    j = i // 16
                    # numA = int(hash[j], 16) % 4
                    # numB = (int(hash[j], 16) >> 2) % 4
                    # RNA.append(numA)
                    # RNA.append(numB)
                    num = (int(hash[j*2:j*2+1], 16) + 1) % 20
                    PepChain.append(num)
            #print(RNA)
            y = self.proteinFolder(PepChain)
            energy = self.MBEPredictor(self.CO2Model, y).detach()[0][0]
            print('Binding Energy: %f' % float(energy))
            if energy > self.diff:
                return True, energy
            else:
                return False, energy
        except Exception as e:
            print('verify PoW: ' + str(e))
            return False, 0

    def verifyBlock(self, block):
        if(len(block) != 1824):
            # Data length is wrong
            return False
        if len(self.chain) == 0:
            previousHash = "0" * 64
        else:
            previousHash = self.chain[-1][1760:1824]
        blockPrivousHash = block[0:64]
        if(blockPrivousHash != previousHash):
            return False
        blockID = block[64:80]
        if(int(blockID) != len(self.chain)):
            # Maybe missing some blocks
            return False
        blockTimestamp = block[80:96]
        if(int(blockTimestamp) > int(time())):
            return False
        blockData = block[96:1056] # 960 bytes
        if not self.verifyTransaction(blockData):
            return False
        blockMiner = block[1056:1248] # 192 bytes
        blockPoWPayload = block[1248:1760] # 512 bytes
        blockHash = block[1760:1824] # 64 bytes
        if(blockHash != sha256(blockPrivousHash + blockID + blockTimestamp + blockData + blockMiner + blockPoWPayload)):
            return False
        return self.verifyPoW(blockPoWPayload, blockHash)[0]

    def makeTransaction(self, blkID, sender, receiver, amount, asset, sk):
        transaction = sender + receiver + str(amount).zfill(64) + asset
        sig = sign(sha256(blkID + transaction), sk)
        return transaction + sig

    def makeWhiteTransaction(self):
        # Only using in Mining
        blockID = str(len(self.chain)).zfill(16)
        return self.makeTransaction(blockID, "0" * 192, "0" * 192, 0, "!" * 320, SigningKey.generate(curve=NIST384p).to_string().hex())

    def makeBlock(self, transaction, miner, pow):
        if len(self.chain) == 0:
            previousHash = "0" * 64
        else:
            previousHash = self.chain[-1][1760:1824]
        blockID = str(len(self.chain)).zfill(16)
        timestamp = str(int(time())).zfill(16)
        block = previousHash + blockID + timestamp + transaction + miner + pow
        return block + sha256(block)

    def makeWhiteMiningReward(self, receiver, pow):
        blockID = str(len(self.chain)).zfill(16)
        whiteTrans = self.makeTransaction(blockID, "0" * 192, "0" * 192, 0, "!" * 320, SigningKey.generate(curve=NIST384p).to_string().hex())
        minedBlock = self.makeBlock(whiteTrans, receiver, pow)
        if self.verifyBlock(minedBlock):
            self.chain.append(minedBlock)
            return True
        return False

def demo():
    sk1 = '9061086a4cb981d215f9875028b6217e812df3a6452abb6df9a23a014a9c40694cb6be00efb817a2ca22393281df8a40'
    vk1 = '75d6c47c5287e9b81fb8472d9ebf6635b5e20f5156d4bacb395997767790120b21fbd9ca67d7eed13ff68e3d8de25bd41ffa661742dcd41a00d83f02bcffd0205f248ea2dcb9432454de53097c78f79e6e0f50f5831ddb8ef4e57e7d1288e82e'
    sk2 = 'd10e1416590b938fc8e57088071af877d81ff20dc39707caa3866894a2a80eb59f249184a6e7f2ac25148082aeadbb24'
    vk2 = '6d731960af08e06ba3e84f660af4af9ccde5d47ba9070602a27687ba6e39b5cb778adcbca6fdef19e99edef115ec6fc9711867ae7e185922596c246f0734640d857c2d7e3f4d87aca6b02f7a891bdccd50ab712af753f0c7d987534bd94fab2b'
    myobj = CO2Core()

    res = myobj.makeWhiteMiningReward(vk1, 'b' * 512)
    if res:
        print("White mining success!")
    else:
        print("White mining failed!")
    # res = myobj.makeWhiteMiningReward(vk1, 'g' * 512)
    # if res:
    #     print("White mining success!")
    # else:
    #     print("White mining failed!")
    print("Balance of vk1: " + str(myobj.checkBalance(vk1)))

    blockID = str(len(myobj.chain)).zfill(16)
    trans = myobj.makeTransaction(blockID, vk1, vk2, 12, '0' * 320, sk1)
    if trans != False:
        block = myobj.makeBlock(trans, vk2, '7' * 512)
        if myobj.verifyBlock(block):
            print("Block is valid")
            myobj.chain.append(block)

    blockID = str(len(myobj.chain)).zfill(16)
    trans = myobj.makeTransaction(blockID, vk1, vk2, 12, '0' * 320, sk1)
    if trans != False:
        block = myobj.makeBlock(trans, vk2, 'i' * 512)
        if myobj.verifyBlock(block):
            print("Block is valid")
            myobj.chain.append(block)
    print("Balance of vk1: " + str(myobj.checkBalance(vk1)))
    print("Balance of vk2: " + str(myobj.checkBalance(vk2)))
    print(myobj.chain)

if __name__ == "__main__":
    demo()
    # listen for new blocks
  
    # Create thread for mining
