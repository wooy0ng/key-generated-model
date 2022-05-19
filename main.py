import argparse
from utils import *
from numpy import ComplexWarning
from mode import *
import warnings

warnings.simplefilter("ignore", ComplexWarning)
# warnings.simplefilter("ignore", UserWarning)
parser = argparse.ArgumentParser()

# path : train, test


# mode : train, test, evaluate, dev
# if you selected evaluate mode, you should set default path.
parser.add_argument('--mode', required=False, default='key_test', help='train, test, key_train, key_test')
parser.add_argument('--path', required=False, default='./dataset/evaluate/')
parser.add_argument('--pretrained_mode', required=False, default=False)

parser.add_argument('--time_limited', required=False, default=5)
parser.add_argument('--down_sampling', required=False, default=False)
parser.add_argument('--down_sampling_rate', required=False, default=16000)

# mfcc param (feature size)
parser.add_argument('--n_mfcc', required=False, default=12)

# window size
parser.add_argument('--windows', required=False, default=5)

# model param
# hidden size : number of latent vector [hidden_size > n_mfcc]
# hidden size : 64, 128, 256
parser.add_argument('--hidden_size', required=False, default=64)     
parser.add_argument('--num_layers', required=False, default=1)

parser.add_argument('--master_key_size', required=False, default=128)
# batch_size 
parser.add_argument('--batch_size', required=False, default=4)

# epoch
parser.add_argument('--epoch', required=False, default=50)

# pretrained
parser.add_argument('--pretrained_model', required=False, default='./model.pt')

args = parser.parse_args()

print(f"[+] {args.mode} mode")
if args.mode == 'train':
    print("[+] train data load...", end=' ')
    train(args)
    key_train(args)
    key_test(args)
elif args.mode == 'key_train':
    print("[+] key train data load...", end=' ')
    key_train(args)
    key_test(args)
elif args.mode == 'test':
    print("[+] test data load...", end=' ')
    test(args)
elif args.mode == 'key_test':
    print("[+] key test data load...", end=' ')
    key_test(args)



