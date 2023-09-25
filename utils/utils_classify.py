import os
import torch
import torch.nn.functional as F

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_output(output, task):

    
    if task == 'normals':
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task == 'classify':

        output=torch.argmax(output.cuda(), dim=1)
        
    elif 'SSL' in task:
        output = output#.permute(0, 2, 3, 1)
    
    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        pass
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output
