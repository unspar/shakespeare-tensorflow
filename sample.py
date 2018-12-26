from __future__ import print_function

from model import Model

def main():
    model = Model(True) #true to sample
    print(model.sample(num=1000))

if __name__ == '__main__':
    main()
