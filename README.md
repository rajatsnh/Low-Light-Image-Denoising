# Low-Light-Image-Denoising
## Description
This project utilizes the Zero-DCE model to implement a low light image enhancement algorithm, achieving a PSNR of 16.664.
The paper used for implementation are :
* https://arxiv.org/pdf/2103.00860
* https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf


Dataset used is the one given in the slack community.


## Implementation Details


* Training Data:The model is trained with images from the train/low and train/high directories.

... * train/low: Contains low light images.

... * train/high: Contains corresponding well-lit images.

* Testing Data:The model processes images from the test/low directory to produce enhanced outputs.

... * test/low: Contains low light images for testing.


