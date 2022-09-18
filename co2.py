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

chain = []
fee = 1
rewardcoins = 128
diff = 0

def checkBalance(address):
    balance = 0
    for block in chain:
        blockData = block[64:1024]
        sender = blockData[0:192]
        receiver = blockData[192:384]
        amount = blockData[384:448]
        if sender == address:
            balance -= (int(amount) + fee)
        if receiver == address:
            balance += int(amount)
    return balance

def checkDoubleSpend(transaction):
    for block in chain:
        blockData = block[64:1024]
        if blockData == transaction:
            return True
    return False

def verifyTransaction(transaction):
    if checkDoubleSpend(transaction):
        return False
    sender = transaction[0:192]
    if sender == "0" * 192:
        # Mining reward, check PoW
        return True
    receiver = transaction[192:384]
    amount = transaction[384:448] # 64 bytes
    asset = transaction[448:768]  # 320 bytes
    if(int(amount) <= 0 or checkBalance(sender) < int(amount)):
        return False
    sig = transaction[768:960]  # 192 bytes
    hash = sha256(sender + receiver + amount + asset)
    return verifySig(hash, sig, sender)

def verifyPoW(PoW):
    return True

def verifyBlock(block):
    if(len(block) != 1600):
        return False
    if len(chain) == 0:
        previousHash = "0" * 64
    else:
        previousHash = chain[-1][1536:1600]
    blockPrivousHash = block[0:64]
    if(blockPrivousHash != previousHash):
        return False
    blockData = block[64:1024]
    if not verifyTransaction(blockData):
        return False
    blockPoW = block[1024:1536]
    blockHash = block[1536:1600]
    if(blockHash != sha256(blockPrivousHash + blockData + blockPoW)):
        return False
    return verifyPoW(blockPoW)

def makeTransaction(sender, receiver, amount, asset, sk):
    if sender != "0" * 192:
        if amount <= 0:
            print("Amount must be positive!")
            return False
        if checkBalance(sender) < amount:
            print("Not enough balance!")
            return False
    transaction = sender + receiver + str(amount).zfill(64) + asset
    sig = sign(sha256(transaction), sk)
    return transaction + sig

def makeBlock(transaction, pow):
    if len(chain) == 0:
        previousHash = "0" * 64
    else:
        previousHash = chain[-1][1536:1600]
    block = previousHash + transaction + pow
    return block + sha256(block)

def makeMiningReward(receiver, pow):
    rewardTrans = makeTransaction("0" * 192, receiver, rewardcoins, "!" * 320, "6" * 96)
    if rewardTrans != False:
        minedBlock = makeBlock(rewardTrans, pow)
        if verifyBlock(minedBlock):
            chain.append(minedBlock)
            return True
    return False

def demo():
    sk1 = '9061086a4cb981d215f9875028b6217e812df3a6452abb6df9a23a014a9c40694cb6be00efb817a2ca22393281df8a40'
    vk1 = '75d6c47c5287e9b81fb8472d9ebf6635b5e20f5156d4bacb395997767790120b21fbd9ca67d7eed13ff68e3d8de25bd41ffa661742dcd41a00d83f02bcffd0205f248ea2dcb9432454de53097c78f79e6e0f50f5831ddb8ef4e57e7d1288e82e'
    sk2 = 'd10e1416590b938fc8e57088071af877d81ff20dc39707caa3866894a2a80eb59f249184a6e7f2ac25148082aeadbb24'
    vk2 = '6d731960af08e06ba3e84f660af4af9ccde5d47ba9070602a27687ba6e39b5cb778adcbca6fdef19e99edef115ec6fc9711867ae7e185922596c246f0734640d857c2d7e3f4d87aca6b02f7a891bdccd50ab712af753f0c7d987534bd94fab2b'
    makeMiningReward(vk1, 'p' * 512)
    makeMiningReward(vk1, 'p' * 512)
    print("Balance of vk1: " + str(checkBalance(vk1)))
    trans = makeTransaction(vk1, vk2, 12, '0' * 320, sk1)
    if trans != False:
        block = makeBlock(trans, 'p' * 512)
        if verifyBlock(block):
            print("Block is valid")
            chain.append(block)
    trans = makeTransaction(vk1, vk2, 12, '0' * 320, sk1)
    if trans != False:
        block = makeBlock(trans, 'p' * 512)
        if verifyBlock(block):
            print("Block is valid")
            chain.append(block)
    print("Balance of vk1: " + str(checkBalance(vk1)))
    print("Balance of vk2: " + str(checkBalance(vk2)))
    print(chain)

if __name__ == "__main__":
    demo()
    # Create thread for mining
