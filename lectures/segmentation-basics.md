# SEGMENTATION BASICS   
## Image segmentation
- process of spatial subsection of a digital image into multiple partitions of pixels according to given criteria
- image simplification
- higher-level object description - regions tend to belong to the same class of object
-- regions may provide object properties - shape, colour, etc
- input for content classifiers - region descriptions can be input for higher classifiers
- over-segmentation - pixels belonging to the same region is classified into the separate regions
- under-segmentation - vice versa
## Concepts of segmentation 
- thresholding methods
-- pixels are categorized based on intensity
- edge-based methods
- region-based methods
-- region growing from seed pixels
-- region splitting and merging for efficient spatial encoding
- clustering and statistical methods
- topographic methods
## Thresholding
- e.g. dark object on a light background
- choose a threshold value, T
- for each pixel - if the brightness at that pixel is less than T it is a pixel of interest otherwise it is part of the background
- use histogram to stipulate regions
## Edge-based segmentation
## Region growing
- split and merge - divide and conquer
- if divided region is homogeneous then no need to split
-- else split further
