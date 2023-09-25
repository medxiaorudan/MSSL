import numpy as np
import torch

def jaccard(gt, pred, void_pixels=None):

    if gt.shape != pred.shape:
        pred = pred[:,1]
    if type(pred)!=np.ndarray and type(gt)!=np.ndarray:
        gt = gt.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        
    assert(gt.shape == pred.shape)
        
    if void_pixels is None:
        void_pixels = np.zeros_like(gt)
    assert(void_pixels.shape == gt.shape)

    gt = gt.astype(np.bool)
    pred = pred.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)
    if np.isclose(np.sum(gt & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(pred & np.logical_not(void_pixels)), 0):
        return 1
    
    else:
        return np.sum(((gt & pred) & np.logical_not(void_pixels))) / \
               np.sum(((gt | pred) & np.logical_not(void_pixels)), dtype=np.float32)

    
def dice_coeff(pred, target):
    smooth = 1.
    
    if type(pred) is np.ndarray:
        pred=torch.from_numpy(pred)
    if type(target) is np.ndarray:
        target=torch.from_numpy(target)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return dice.tolist()
 

def iou_mean(pred, target, n_classes = 1):
#n_classes ：the number of classes in your dataset,not including background
# for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)
    print("pred:",pred.shape)
    print("target",target.shape)
    print("pred:",pred)
    print("target",target)
  # Ignore IoU for background class ("0")
    for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        print("pred_inds:",pred_inds)
        target_inds = target == cls
        print("target_inds:",target_inds)
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        print("intersection:",intersection)
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        print("union:",union)
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum/n_classes


def precision_recall(gt, pred, void_pixels=None):
    if gt.shape != pred.shape:
        pred = pred[:,1]
    if type(pred)!=np.ndarray and type(gt)!=np.ndarray:
        gt = gt.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
#    assert(gt.shape == pred.shape)
    if void_pixels is None:
        void_pixels = np.zeros_like(gt)

    gt = gt.astype(np.bool)
    pred = pred.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)

    tp = ((pred & gt) & ~void_pixels).sum()
    fn = ((~pred & gt) & ~void_pixels).sum()

    fp = ((pred & ~gt) & ~void_pixels).sum()

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)

    return prec, rec
