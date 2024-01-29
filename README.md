# ComfyUI-DareMerge
Merge two checkpoint models by dare ties (https://github.com/yule-BUAA/MergeLM).  Now with CLIP support and a bunch of other stuff.

Check out Noise Injection for a fun time, and the LoRA loader that can read the tags.

# Node List

## U-Net
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|unet|Model Merger (Advanced)|`MODEL`, `MODEL`, `LAYER_GRADIENT`, `MODEL_MASK (optional)`|`MODEL`|Performs a model merge, with gradient configuration for layer weights|
|unet|Model Merger (Advanced/DARE)|`MODEL`, `MODEL`, `LAYER_GRADIENT`, `MODEL_MASK (optional)`|`MODEL`|Performs a DARE-TIES merge (using layers is for targeted control)|
| --- | --- | --- | --- | --- |
|unet|Model Merger (Block)|`MODEL`, `MODEL`, `MODEL_MASK (optional)`|`MODEL`|Performs a block merge|
|unet|Model Merger (Block/DARE)|`MODEL`, `MODEL`, `MODEL_MASK (optional)`|`MODEL`|Performs a DARE block merge|
|unet|Model Merger (MBW/DARE)|`MODEL`, `MODEL`, `MODEL_MASK (optional)`|`MODEL`|Performs a DARE block merge (using MBW)|
|unet|Model Merger (Attention/DARE)|`MODEL`, `MODEL`, `MODEL_MASK (optional)`|`MODEL`|Performs a DARE block merge (targeting attention)|

## Layer Gradient
Gradients control the merge ratios for layers.
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|grad|Gradient Operations|`LAYER_GRADIENT`, `LAYER_GRADIENT`|`LAYER_GRADIENT`|Performs operations on layer gradients|
|grad|Gradient Edit|`LAYER_GRADIENT`|`LAYER_GRADIENT`|Directly target layers for editing with wildcards|
|grad|Block Gradient|`MODEL`|`LAYER_GRADIENT`|Returns the block gradient for a model|
|grad|Attention Gradient|`MODEL`|`LAYER_GRADIENT`|Returns the attention gradient for a model|
|grad|Shell Gradient|`MODEL`|`LAYER_GRADIENT`|Returns the balanced layers (onion) gradient for a model|
|grad|MBW Gradient|`MODEL`|`LAYER_GRADIENT`|Returns the MBW-style gradient for a model|

### Merging
* In general, one means keep first model, zero means keep second model
* For DARE, we use the base model to determine which values to protect or include.
* For TIES, we can use the sum of the delta ties (as in the paper), or the count, or off to disable.
* Can accept a model mask, which will restrict changes to only modify the masked areas.

## Masking
These are masks that whitelist or blacklist parameters for a merge.  They are used to filter parameters that are wanted in a merge, and can be used to protect or target the parameters of a model for changes.
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|mask|Simple Masker|`MODEL`|`MODEL_MASK`|Creates a new mask for a model|
|mask|Magnitude Masker|`MODEL`, `MODEL`|`MODEL_MASK`|Creates a mask based on the deltas of the parameters|
|mask|Quad Masker|`MODEL`|`MODEL_MASK`, `MODEL_MASK`, `MODEL_MASK`, `MODEL_MASK`|Creates four random non-overlapping masks|
|mask|Mask Operations|`MODEL_MASK`, `MODEL_MASK`|`MODEL_MASK`|Allows set operations to be performed on masks|
|mask|Mask Edit|`MODEL_MASK`|`MODEL_MASK`|Allows the direct editing of mask layers|

## CLIP
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|clip|CLIP Merger (DARE)|`CLIP`, `CLIP`|`CLIP`|Performs a DARE merge on two CLIP|

## LoRA
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|lora|LoRA Loader (Tags)|`MODEL`, `CLIP`|`MODEL`, `CLIP`, `STRING`|Loads a LoRA model, returning the tags from the metadata|

## Utilities
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|util|Normalize Model|`MODEL`, `MODEL`|`MODEL`|Normalizes one models parameter norm to another model|
|util|Inject Noise|`MODEL`|`MODEL`|Injects noise in to a model|

## Reporting
|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|report|Mask Reporting|`MODEL_MASK`|`STRING`, `IMAGE`|Returns basic layer statistics for the mask|
|report|Model Reporting|`MODEL`|`STRING`, `IMAGE`|Returns a plot of a model layer|
|report|LoRA Reporting||`STRING`, `IMAGE`|Returns stats and information about a LoRA|
|report|Gradient Reporting|`LAYER_GRADIENT`|`STRING`, `IMAGE`|Returns a report on the layer gradient|

## How to use
DARE-TIES does a stochastic selection of the parameters to keep, and then only performs updates in either the 'up' or 'down' direction.  According to the paper, the workflow should be as such:
* Take our model A, and build a magnitude mask based on a base model.
* Take model B, and merge it in to model A using the mask to protect model A's largest parameters.

*Of note, this merge method does use random sampling, so you should not just assume that your first random seed is the best one for your merge, and if it is not set to fixed that the merge will change every run.*

### Layer Gradients
Layer gradients are a generalized term to describe the merge ratios for a model merge.  These define the ratios at each layer of a model, which allows model ratio selection to become a more fine-grained operation which operations can be performed on.  I have left some basic components which do not use LAYER_GRADIENT, but there is nothing special about them as internally they are just using the advanced components.

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

### Mask Editing
Selecting which parameters by magnitude masking may be one approach, but a potentially more powerful approach would be pure random selection; until we can find a pattern in our latent space and account for it, we can assume that the distribution of the parameters for a given state is random.  The mask editor can target individual layers of the mask, and generate a boolean or random (bernoulli or gaussian) mask for that layer.  * is a wildcard and will match all layers.  If you need to find the layer names, you can see them in the mask reporting node 'details' report.

### Normalization
I am testing out a new normalization method, which is to normalize the norm of the parameters of one model to another.  This is done by taking the ratio of the norms, and then scaling the parameters of the first model by that ratio.  This is done in the `Normalize Model` node.  There are a few options, of most interest is the 'attn_only' option, which only scales Q and K relative to each other, and just that.  You should see no difference in the model's performance, but it might make the merge more stable.

### Noise Injection
Noise injection is a really fun tool that you can use to inject noise into targeted layers (or parts of layers, using a model mask), in a manner similar to https://github.com/EGjoni/DRUGS.

### LoRA Loader (Tags)
This node will load a LoRA model, and return the tags from the metadata as a comma separated string, ready to pass to some other node.