import argparse

# thanks Maxim from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    if '[' in v and ']' in v:
        return list(map(int, v.strip('[]').split(',')))
    else:
        raise argparse.ArgumentTypeError('Input expected in the form [b1,b2,b3,...]')

def scheduler_type_choice(v):
    if v == 'fixed' or v == 'annealing' or v == 'custom_warmup_anneal' or v == 'noam' or v == 'noam_augmented' or v == 'cosine':
        return v
    else:
        raise argparse.ArgumentTypeError('Argument must be either \'fixed\', '
                                         '\'annealing\', '
                                         '\'cosine\','
                                         '\'custom_warmup_anneal\', '
                                         '\'noam\'.')