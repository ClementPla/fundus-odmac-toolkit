![header](imgs/header.png)


# The Fundus ODMAC Toolkit

This project aims to evaluate the performance of different models for the segmentation of optic disk and macula in fundus images. The reported perfomance metrics are not always consistent in the literature. Our goal is to provide a fair comparison between different models using the same datasets and evaluation protocol.


## Installation

```bash
pip install .
```

or
```bash
pip install -e .
```

## Benchmarck

### IDRiD Segmentation Challenge (Optic Disk)

| Architecture 	|        Encoder        	| Jaccard Index 	|
|:------------:	|:---------------------:	|:-------------:	|
| UNet         	| seresnet50            	| 92.74%        	|
| UNet         	| mobilevitv2_100       	| 92.24%        	|
| UNet         	| mobilenetv3_small_050 	| 84.29%        	|
| UNet         	| maxvit_tiny_tf_512    	| 91.80%        	|
| UNet         	| maxvit_small_tf_512   	| 93.17%        	|
| UNet         	| maxvit_base_tf_512    	| 87.20%        	|
| UNet++       	| seresnet50            	| 90.75%        	|
| UNet++       	| mobilenetv3_small_050 	| 86.04%        	|