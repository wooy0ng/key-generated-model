import torch
import pickle as pkl


def save_model(model, name='obj'):
    print("[+] model save mode", end=' ')
    torch.save(model.state_dict(), f'./{name}.pt')
    print("[complete]")

def load_model(args, model, name='obj'):
    print("[+] model load mode", end=' ')
    path = args.pretrained_model
    model.load_state_dict(torch.load(f'./{name}.pt'))
    print("[complete]")
    return model

def save_object(object, name='obj'):
    print("[+] object save mode", end=' ')
    pkl.dump(object, open(f"{name}.pkl", "wb+"))
    print("[complete]")


def load_object(args, name='obj'):
    print("[+] object load mode", end=' ')
    with open(f"{name}.pkl", "rb+") as obj:
        _object = pkl.load(obj)
    
    print("[complete]")
    return _object
    