import sys
import dataloaders.loader


def main(config_paths):
    data = dataloaders.loader.load_train()
    print(data.classes)


if __name__ == '__main__':
    main(sys.argv[1:])
