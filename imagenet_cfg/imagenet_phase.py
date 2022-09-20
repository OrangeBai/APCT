phase = {
    'phase_0':
        {
            'lr_scheduler': 'linear',
            'lr': 0.,
            'lr_e': 0.01,
            'batch_size': 128,
            'data_size': 160,
            'crop_size': 128,
            'start_epoch': 0,
            'num_epoch': 1,
        },
    'phase_1':
        {
            'lr_scheduler': 'linear',
            'lr': 0.1,
            'lr_e': 0.01,
            'batch_size': 128,
            'data_size': 160,
            'crop_size': 128,
            'start_epoch': 1,
            'end_epoch': 5,
        },
    'phase_2':
        {
            'lr_scheduler': 'linear',
            'lr': 0.1,
            'lr_e': 0.01,
            'batch_size': 128,
            'data_size': 160,
            'crop_size': 128,
            'start_epoch': 5,
            'num_epoch': 10,
        },
    'phase_3':
        {
            'lr_scheduler': 'linear',
            'lr': 0.1,
            'lr_e': 0.01,
            'batch_size': 128,
            'data_size': 160,
            'crop_size': 128,
            'start_epoch': 1,
            'num_epoch': 5,
        }
}
