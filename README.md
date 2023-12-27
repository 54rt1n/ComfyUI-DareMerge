# ComfyUI-DareMerge
Merge two checkpoint models by dare ties (https://github.com/yule-BUAA/MergeLM), sort of.

# Node List

## DareModelMerger

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|merge|DareModelMerger|`MODEL`, `MODEL`|`MODEL`|(input, middle, out, sparsity)|

* input, middle, and out are the block region weights.
* sparsity is the level to sample the peak differences between the two models.  Zero applies nothing, 1 applies the full normal block weights for the second model.


