
import os
import torch
from datetime import datetime

from torch.nn.parameter import Parameter


def load_most_recent_checkpoint(model, optimizer, data_loader, rank,
                                save_model_path, datetime_format='%Y-%m-%d-%H:%M:%S'):
    ls_files = os.listdir(save_model_path)
    # search for the most recent checkpoint
    most_recent_checkpoint_datetime = None
    most_recent_checkpoint_filename = None
    most_recent_checkpoint_info = 'no_additional_info'
    for file_name in ls_files:
        if file_name.startswith('checkpoint_'):
            _, datetime_str, _, info, _ = file_name.split('_')
            # search for every path that starts with 'checkpoint'
            file_datetime = datetime.strptime(datetime_str, datetime_format)
            if (most_recent_checkpoint_datetime is None) or \
                    (most_recent_checkpoint_datetime is not None and
                     file_datetime > most_recent_checkpoint_datetime):
                most_recent_checkpoint_datetime = file_datetime
                most_recent_checkpoint_filename = file_name
                most_recent_checkpoint_info = info

    if most_recent_checkpoint_filename is not None:
        print("Loading: " + str(save_model_path + most_recent_checkpoint_filename))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(save_model_path + most_recent_checkpoint_filename,
                                map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if data_loader is not None:
            data_loader.set_num_epoch(checkpoint['data_loader_num_epoch'])
        if data_loader is not None:
            data_loader.set_batch_it(checkpoint['data_loader_batch_it'])
        model.trained_steps = int(checkpoint['model_trained_steps'])
        return True, most_recent_checkpoint_info
    else:
        print("Loading: no checkpoint found in " + str(save_model_path))
        return False, most_recent_checkpoint_info


def save_last_checkpoint(model, optimizer, data_loader, save_model_path,
                         num_max_checkpoints=3, datetime_format='%Y-%m-%d-%H:%M:%S',
                         additional_info='noinfo'):
    if num_max_checkpoints == 0:
        return

    checkpoint = {
        'model_trained_steps': model.trained_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'data_loader_num_epoch': data_loader.get_num_epoch(),
        'data_loader_batch_it': data_loader.get_batch_it()}

    ls_files = os.listdir(save_model_path)
    # search for the most recent checkpoint
    oldest_checkpoint_datetime = None
    oldest_checkpoint_filename = None
    num_check_points = 0
    for file_name in ls_files:
        if file_name.startswith('checkpoint_'):
            num_check_points += 1
            _, datetime_str, _, _, _ = file_name.split('_')
            # search for every path that starts with 'checkpoint'
            file_datetime = datetime.strptime(datetime_str, datetime_format)
            if (oldest_checkpoint_datetime is None) or \
                    (oldest_checkpoint_datetime is not None and file_datetime < oldest_checkpoint_datetime):
                oldest_checkpoint_datetime = file_datetime
                oldest_checkpoint_filename = file_name

    if oldest_checkpoint_filename is not None and num_check_points == num_max_checkpoints:
        os.remove(save_model_path + oldest_checkpoint_filename)  # remove the last checkpoint file

    # create a new one
    new_checkpoint_filename = 'checkpoint_' + datetime.now().strftime(datetime_format) + '_' + str(
        checkpoint['model_trained_steps']) + '_' + str(additional_info) + '_.pth'
    print("Saved to " + str(new_checkpoint_filename))
    torch.save(checkpoint, save_model_path + new_checkpoint_filename)


def partially_load_state_dict(model, state_dict, verbose=False):
    own_state = model.state_dict()
    num_print = 999
    count_print = 0
    for name, param in state_dict.items():
        if name not in own_state:
            if verbose:
                print("Not found: " + str(name))
            continue
        if isinstance(param, Parameter):
            param = param.data

        if verbose:
            if count_print < num_print:
                print("Found: " + str(name))
                count_print += 1
        own_state[name].copy_(param)