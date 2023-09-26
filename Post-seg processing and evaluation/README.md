## Scripts in this Subfolder

- `mask_skeleton.m`: Extracts the low-resolution image (`level3.png`) from the WSI.
- `overlay.ipynb`: Essentially does the same as the above but at different levels of the WSI pyramid.
- `soverlays.m`: Performs subsampling of the WSI image based on the `mask_gt.png` image and saves all the subimages in a `subimages` folder. It also creates the `subimages`, `skeletons`, and `overlays` folders.
- `predict_clean_smooth.m`: Segments the tumor areas of the tissue with unsupervised learning. It aims to be an automated version of `segment_roi_gt.py` but requires improvements.
- `rescale01.m`: Extracts the low-resolution image (`level3.png`) from the WSI.
- `vscular_features_classification.ipynb`: Essentially does the same as the above but at different levels of the WSI pyramid.
- `Xavier_evaluation.m`: Performs subsampling of the WSI image based on the `mask_gt.png` image and saves all the subimages in a `subimages` folder. It also creates the `subimages`, `skeletons`, and `overlays` folders.
- `xavier_evaluation_python_code.ipynb`: Segments the tumor areas of the tissue with unsupervised learning. It aims to be an automated version of `segment_roi_gt.py` but requires improvements.

## The proposed evaluation method

We proposed a new evaluation function to restrict the results in terms of vessel detection, basically considering the length but not the width of vessels. We dilated (with a disk of radius 3, according to experiment Table `tab2`) the segmentation result from \( S \) to obtain \( DS \) and the ground truth \( GT \) to obtain \( DGT \). We computed the ratio of miss-detected vessels as:

$$
MV = \frac{|\{(i,j): GT(i,j) = 1, DS(i,j) = 0\}|}{|\{(i,j): GT(i,j) = 1\}|}
$$

and the ratio of falsely detected vessels as:

$$
FV = \frac{|\{(i,j): S(i,j) = 1, DGT(i,j) = 0\}|}{|\{(i,j): S(i,j) = 1\}|}
$$

Finally, we defined the following global performance index:

$$
IV = 1 - \frac{MV+VF}{2}
$$
