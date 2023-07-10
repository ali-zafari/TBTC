class Config:

    class Model:
        model = 'zyc2022-swint-charm'  # choose one from compressai.zoo.image_models
        quality = 'M'
        lmbda = 0.01
        beta = None

    class Data:
        dataset_directory = 'path/to/dataset'

        train_batch_size = 8
        valid_batch_size = 1

        train_patch_size = (256, 256)
        valid_patch_size = (1024, 1024)

        num_workers = 8
        pin_memory = True
        persistent_workers = False

    class Trainer:
        accelerator = 'gpu'
        devices = [1, ]  # list of devices to train on

        num_epochs = 5000
        seed = 10
        net_lr = 0.0001
        aux_lr = 0.0001

        validation_cadence = 10  # [epochs]
        gradient_clip_norm = 10.0  # necessary for gradient-exploding-free training

        checkpoint_to_resume = None  # 'path/to/a/checkpoint'
        log_cadence = 25  # [steps] Note: global_steps=accumulated total number of calling step() for any optimizer
                          # but this value gets compared to the number of training_steps which is counted only once for
                          # each usage of training_step
        
        strategy = 'auto'  # "ddp_find_unused_parameters_true" or "auto"
        float32_matmul_precision = None
