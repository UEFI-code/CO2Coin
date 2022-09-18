import socket
import co2

nodelist = []

def listener():
    s = socket.socket()
    host = socket.gethostname()
    port = 2333
    s.bind((host, port))
    s.listen()
    while True:
        c, addr = s.accept()
        nodelist.append(addr)
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
            if block not in co2.chain and co2.verifyBlock(block):
                co2.chain.append(block)
                for i in range(len(co2.jobs)):
                    if co2.jobs[i] == block[64:1024]:
                        co2.jobs.pop(i)
                        break
                c.send(b'Verified!')
            else:
                c.send(b'Error block!')    
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