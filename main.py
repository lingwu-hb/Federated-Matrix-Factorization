import torch.cuda
import torch
from parse import get_parse
from utils import fix_seed
from FRS import Clients, Server
from centralizedMF import centralizedMF

def main():
    fix_seed()
    args = get_parse()

    # construct clients
    clients = Clients(args)
    # construct the server
    server = Server(args, clients)
    server.train()

def cenMFmain():
    fix_seed()
    args = get_parse()

    cenMF = centralizedMF(args)
    cenMF.train()

if __name__ == '__main__':
    cenMFmain()

