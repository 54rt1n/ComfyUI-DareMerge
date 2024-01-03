# ComfyUI-DareMerge
Merge two checkpoint models by dare ties (https://github.com/yule-BUAA/MergeLM), sort of.

# Node List

## DareModelMerger

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|merge|DareModelMerger|`MODEL`, `MODEL`, `MODEL`|`MODEL`|Performs a DARE block merge|
|merge|MagnitudeModelMerger|`MODEL`, `MODEL`|`MODEL`|Performs a MP block merge|
|merge|BlockModelMergerAdv|`MODEL`, `MODEL`|`MODEL`|Performs a block merge with a custom merge type|

* In general, one means keep first model, zero means keep second model
* For Dare, we use the base model to determine which values to protect or include.  It is optional, and if not provided will ignore exclude_a, include_b, threshold_type, and invert.
* Larger density means to include more of the second model
* Larger exclude_a preserves more of the first model
* Larger include_b allows more of the second model
* input, middle, and out are the block region weights.
* threshold_type is the way we determine where our threshold lies in our distribution since we use chunks, quantile will err towards the sparsity, median will use the chunk median.
* invert is whether we invert the threshold, so we keep the weights that are below the threshold instead of above.


