from prepare_data import training_path

fold_number_to_val_instrument_numbers  = {0: [1],
        1: [2],
        2: [3],
        3: [4]}

num_folds = len(fold_number_to_val_instrument_numbers)

def get_split(fold):
    val_file_names = []
    train_file_names = []

    for instrument_number in range(1, 4):
        if instrument_number in fold_number_to_val_instrument_numbers[fold]:
            val_file_names += list((training_path / f'instrument{instrument_number}' / 'validation' / 'frames').glob('*'))
        else:
            train_file_names += list((training_path / f'instrument{instrument_number}' / 'training' / 'frames').glob('*'))
    
    assert len(train_file_names) > 0 and len(val_file_names) > 0, 'Required to have training and validation files for a fold.'
    
    return train_file_names, val_file_names
