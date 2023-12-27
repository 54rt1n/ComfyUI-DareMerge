# ComfyUI-DareMerge
Merge two checkpoint models by dare ties (https://github.com/yule-BUAA/MergeLM), sort of.

# Node List

## DareModelMerger

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|merge|DareModelMerger|`MODEL`, `MODEL`|`MODEL`|Performs a DARE block merge|

* input, middle, and out are the block region weights.
* sparsity is the level to sample the peak differences between the two models.  Zero applies nothing, 1 applies the full normal block weights for the second model.
* threshold_type is the way we determine where our threshold lies in our distribution since we use chunks, quantile will err towards the sparsity, median will use the chunk median.
* invert is whether we invert the threshold, so we keep the weights that are below the threshold instead of above.


