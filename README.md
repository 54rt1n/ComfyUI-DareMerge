# ComfyUI-DareMerge
Merge two checkpoint models by dare ties (https://github.com/yule-BUAA/MergeLM), sort of.

# Node List

## DareModelMerger

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|merge|DareModelMerger|`MODEL`, `MODEL`, `MODEL`|`MODEL`|Performs a DARE block merge|
|merge|MagnitudePruningModelMerger|`MODEL`, `MODEL`|`MODEL`|Performs a MP block merge|

* In general, one means keep first model, zero means keep second model
* For Dare, we use the base model to determine which values to protect or include.
* Larger sparsity means to discard more of the second model
* input, middle, and out are the block region weights.
* threshold_type is the way we determine where our threshold lies in our distribution since we use chunks, quantile will err towards the sparsity, median will use the chunk median.
* invert is whether we invert the threshold, so we keep the weights that are below the threshold instead of above.


