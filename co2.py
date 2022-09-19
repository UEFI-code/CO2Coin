from ecdsa import SigningKey, VerifyingKey, NIST384p
import hashlib

def sha256(msg):
    return hashlib.sha256(msg.encode()).hexdigest()

def generate_keys():
    sk = SigningKey.generate(curve=NIST384p)
    vk = sk.get_verifying_key()
    return sk.to_string().hex(), vk.to_string().hex()

def sign(msg, sk):
    return SigningKey.from_string(bytes.fromhex(sk), curve=NIST384p).sign(msg.encode()).hex()

def verifySig(msg, sig, vk):
    return VerifyingKey.from_string(bytes.fromhex(vk), curve=NIST384p).verify(bytes.fromhex(sig), msg.encode())

class CO2Core:
    def __init__(self):
        self.chain = []
        self.jobs = []
        self.fee = 1
        self.rewardcoins = 128
        self.diff = 0

    def checkBalance(self, address):
        balance = 0
        for block in self.chain:
            blockData = block[64:1024]
            sender = blockData[0:192]
            receiver = blockData[192:384]
            amount = blockData[384:448]
            miner = block[1024:1216]
            if sender == address:
                balance -= (int(amount) + self.fee)
            if receiver == address:
                balance += int(amount)
            if miner == address:
                balance += self.rewardcoins
        return balance

    def checkDoubleSpend(self, transaction):
        for block in self.chain:
            blockData = block[64:1024]
            if blockData == transaction:
                return True
        return False

    def verifyTransaction(self, transaction):
        if self.checkDoubleSpend(transaction):
            return False
        sender = transaction[0:192]
        receiver = transaction[192:384]
        if sender == "0" * 192 and receiver == "0" * 192:
            if(len(self.jobs) == 0):
                print("No jobs to verify, but may give you white mined reward!")
                return True
            print('Pls verify blocks, don\'t do white mining!')
            return False
        amount = transaction[384:448] # 64 bytes
        asset = transaction[448:768]  # 320 bytes
        if(int(amount) <= 0 or (self.checkBalance(sender) - self.fee) < int(amount)):
            return False
        sig = transaction[768:960]  # 192 bytes
        hash = sha256(sender + receiver + amount + asset)
        return verifySig(hash, sig, sender)

    def verifyPoW(self, PoW):
        if len(PoW) != 512:
            return False
        for i in self.chain:
            if i[1216:1728] == PoW:
                print("Not your PoW!")
                return False
        return True

    def verifyBlock(self, block):
        if(len(block) != 1792):
            # Data length is wrong
            return False
        if len(self.chain) == 0:
            previousHash = "0" * 64
        else:
            previousHash = self.chain[-1][1728:1792]
        blockPrivousHash = block[0:64]
        if(blockPrivousHash != previousHash):
            return False
        blockData = block[64:1024]
        if not self.verifyTransaction(blockData):
            return False
        blockMiner = block[1024:1216] # 192 bytes
        blockPoW = block[1216:1728] # 512 bytes
        blockHash = block[1728:1792] # 64 bytes
        if(blockHash != sha256(blockPrivousHash + blockData + blockMiner + blockPoW)):
            return False
        return self.verifyPoW(blockPoW)

    def makeTransaction(self, sender, receiver, amount, asset, sk):
        transaction = sender + receiver + str(amount).zfill(64) + asset
        sig = sign(sha256(transaction), sk)
        return transaction + sig

    def makeBlock(self, transaction, miner, pow):
        if len(self.chain) == 0:
            previousHash = "0" * 64
        else:
            previousHash = self.chain[-1][1728:1792]
        block = previousHash + transaction + miner + pow
        return block + sha256(block)

    def makeWhiteMiningReward(self, receiver, pow):
        whiteTrans = self.makeTransaction("0" * 192, "0" * 192, 0, "!" * 320, SigningKey.generate(curve=NIST384p).to_string().hex())
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

    res = myobj.makeWhiteMiningReward(vk1, 'p' * 512)
    if res:
        print("White mining success!")
    else:
        print("White mining failed!")
    res = myobj.makeWhiteMiningReward(vk1, 'p' * 512)
    if res:
        print("White mining success!")
    else:
        print("White mining failed!")

    print("Balance of vk1: " + str(myobj.checkBalance(vk1)))
    trans = myobj.makeTransaction(vk1, vk2, 12, '0' * 320, sk1)
    if trans != False:
        block = myobj.makeBlock(trans, vk2, 'p' * 512)
        if myobj.verifyBlock(block):
            print("Block is valid")
            myobj.chain.append(block)
    trans = myobj.makeTransaction(vk1, vk2, 12, '0' * 320, sk1)
    if trans != False:
        block = myobj.makeBlock(trans, vk2, 'p' * 512)
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
