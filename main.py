from models import Model

def main():
    print('Running main')
    model = Model('ssd', 2, 0)
    model.train('./conf/torch.yaml', 2, workers=2)


if __name__ == '__main__':
    main()