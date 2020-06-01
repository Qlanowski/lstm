import sys
import torch.utils.data


def main(config_paths):
    for config_path in config_paths:
        with open(config_path, 'r') as file:
            json_config = file.read()

        
        #     config = tester.TestConfig.from_json(json_config)
        # print(f"Test {config.test_name} started")
        # #network = tester.perform_test(config)
        # network = config.network
        # network.load_state_dict(torch.load("C:\\Users\\ulanowskik\\Desktop\\cnn\\test_results\\resnet_nofreeze_all_test\\test00_model.pth"))
        # print("Preparing submission...")
        # sub_file = f'kaggle_submissions/{config.test_name}.csv'
        # kaggle_testset = datasets.cifar10.from_kaggle(train=False, input_size=config.input_size)
        # predicted = kaggle.predict(
        #     network=network,
        #     set_loader=torch.utils.data.DataLoader(
        #         dataset=kaggle_testset,
        #         batch_size=100),
        # )
        # submission_df = kaggle.create_submission_df(predicted, idx_to_class)
        # submission_df.to_csv(sub_file, index=False)
        # print(f'Submission saved to {sub_file}')


if __name__ == '__main__':
    main(sys.argv[1:])
