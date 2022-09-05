from cli import get_args
from data import prepare_data

if __name__=='__main__':
    args = get_args()
    train, test = prepare_data(args)
    