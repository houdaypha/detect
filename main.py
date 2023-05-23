from models import Model

def main():
    model = Model('fasterrcnn', 1, 'gpu')
    model.train('./conf/torch.yaml', 2, workers=2)


if __name__ == '__main__':
    main()