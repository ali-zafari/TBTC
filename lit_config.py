class Config:

    class Model:
        model = 'bmshj2018-factorized'  # choose one from compressai.zoo.image_models
        quality = 1
        lmbda = 0.01
        beta = None

    class Data:
        dataset_directory = '/mnt/abbb7b18-d9ab-4b62-98ca-c90e47f31d4d/CLICtrainCLICvalidKodaktest'

        train_batch_size = 4
        valid_batch_size = 1

        train_patch_size = (256, 256)
        valid_patch_size = (1024, 1024)

        num_workers = 8
        pin_memory = True
        persistent_workers = False

    class Trainer:
        accelerator = 'gpu'
        devices = [1, ]  # list of devices to train on

        num_epochs = 2
        seed = 10
        net_lr = 0.0001
        aux_lr = 0.0001

        validation_cadence = 1  # [epochs]
        gradient_clip_norm = 10.0  # necessary for gradient-exploding-free training

        checkpoint_to_resume = None  # 'path/to/a/checkpoint'
        log_cadence = 25  # [steps] Note: global_steps=accumulated total number of calling step() for any optimizer
                          # but this value gets compared to the number of training_steps which is counted only once for
                          # each usage of training_step
        strategy = 'auto'  # "ddp_find_unused_parameters_true" or "auto"
        float32_matmul_precision = None
