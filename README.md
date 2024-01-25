# ComfyUI-DareMerge
Merge two checkpoint models by dare ties (https://github.com/yule-BUAA/MergeLM).  Now with CLIP support.

# Node List

## U-Net
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|unet|Model Merger (Masked)|`MODEL`, `MODEL`, `MODEL_MASK`|`MODEL`|Performs a masked block merge|
|unet|Model Merger (DARE)|`MODEL`, `MODEL`, `MODEL_MASK (optional)`|`MODEL`|Performs a DARE block merge|
|unet|MBW Merger (DARE)|`MODEL`, `MODEL`, `MODEL_MASK (optional)`|`MODEL`|Performs a DARE block merge, with full layer control (like MBW)|

## CLIP
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|clip|CLIP Merger (DARE)|`CLIP`, `CLIP`|`CLIP`|Performs a DARE merge on two CLIP|

## Masking
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|mask|Magnitude Masker|`MODEL`, `MODEL`|`MODEL_MASK`|Creates a mask based on the deltas of the parameters|
|mask|Mask Operations|`MODEL_MASK`, `MODEL_MASK`|`MODEL_MASK`|Allows set operations to be performed on masks|

## Utilities
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|util|Normalize Model|`MODEL`, `MODEL`|`MODEL`|Normalizes one models parameter norm to another model|

### Merging
* In general, one means keep first model, zero means keep second model
* For DARE, we use the base model to determine which values to protect or include.
* For TIES, we can use the sum of the delta ties (as in the paper), or the count, or off to disable.
* Can accept a model mask, which will restrict changes to only modify the masked areas.

## How to use
DARE-TIES does a stochastic selection of the parameters to keep, and then only performs updates in either the 'up' or 'down' direction.  According to the paper, the workflow should be as such:
* Take our model A, and build a magnitude mask based on a base model.
* Take model B, and merge it in to model A using the mask to protect model A's largest parameters.

*Of note, this merge method does use random sampling, so you should not just assume that your first random seed is the best one for your merge, and if it is not set to fixed that the merge will change every run.*

### Masks
* A mask creates a signature of a model to filter parts of a model merge.  Areas that are selected in the mask will be included in the merge, while areas that are not selected will be excluded.
* Using the mask you can select the parameters to target either stronger, by using the 'above' or weaker, by using the 'below' option.  The threshold will determine quantile of the parameters to target.
* Since these tensors are large they have to be chunked, which is where the thredhold type comes in as it is used to determine the set level of the threshold.  The options 'median' and 'quantile' refer to how to handle the chunks.  'median' will err more towards the middle, while 'quantile' will err more towards the edges.
* Training (and overtraining) will generate high magnitide parameters.  Different models have high strength parameters in different parts of their model, so filtering out high magnitude parameters (by selecting 'below', with model A as your filter target and SD1.5 as the base) will allow a merge to not disturb the high strength parameters of model A.  There is an example of this in the examples folder.

### Set Operations
* The set operations are performed on the masks, and are used to combine masks together.  The operations are:
  * Union: Selects the union of the two masks
  * Intersection: Selects the intersection of the two masks
  * Difference: Subtracts the second mask from the first
  * Xor: Selects the symmetric difference of the two masks - the union minus the intersection

#### What does this mean?
I don't know if it means you could potentially bin the parameters of a model up with some fancy set thresholding, and then merge a different model in to each slice...  It might mean that though; and probably does.

### Normalization
I am testing out a new normalization method, which is to normalize the norm of the parameters of one model to another.  This is done by taking the ratio of the norms, and then scaling the parameters of the first model by that ratio.  This is done in the `Normalize Model` node.  There are a few options, of most interest is the 'attn_only' option, which only scales Q and K relative to each other, and just that.  You should see no difference in the model's performance, but it might make the merge more stable.
