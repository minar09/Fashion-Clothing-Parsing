# CFPD | Colorful Fashion Parsing Data

This dataset is used in the paper, [(S. Liu, J. Feng, C. Domokos, H. Xu, J. Huang, Z. Hu, & S. Yan. 2014) CFPD | Fashion parsing with weak color-category labels.](https://sites.google.com/site/fashionparsing/home)

## Details

- 2,682 images
- 600 x 400 (height, width)
- pixel-level annotated (segmentation map)
	- 23 categories
	- 13 **colors**
- `make_label.py` makes the followings from `fashon_parsing_data.mat`.
	- `label/bbox.json`: **bounding box** (for object detection not semantic segmentation).
	- `label/categories.tsv`
		- `category_id`
		- `category`
- `label/main_categories.tsv` is selected from `categories.tsv` for object detection.

## Setup Dataset

```
# install the requirements
pip install -r requirements.txt

# download zip file (name) from author's GoogleDrive
download.sh

# rename "data" file manually to "data.zip" or run the command below:
rename data data.zip

# unzip data.zip manually or follow instructions from here (http://stahlworks.com/dev/index.php?tool=zipunzip) to unzip from cmd

# make label/categories.tsv
# make label/bbox.json
# from fashon_parsing_data.mat
python make_label.py
```

## fashon_parsing_data.mat

### structure

- #refs#
	- 0, 0A~zz: 2,719 groups (each record may be a image info)
		- category_label
			- np.array, float64(actually int), (1, 425)
			- map: super-pix id (1~425) -> fine category id (1~117)
		- color_label
			- np.array, float64(actually int), (1,425)
			- map: super-pix id (1~425) -> fine color id (1~60)
		- ?img_name
			- np.array, uint16, (2~9 etc, 1)
			- range: 59~108 etc
			- ???
		- segmentation
			- np.array, float32, (400, 600)
			- map: img pix loc (w, h) -> super-pix id (1~425)
			- (height, width) (transposing this) is the right orientation.
	- b~x: 23 datasets (each record means category id)
		- np.array, uint16, (1, #fine_category)
		- range: 1~117
		- fine categories's ids under the cateogry.
	- y,z,A~K: 13 datasets (each record means color id)
		- np.array, uint16, (1, #fine_color)
		- range: 1~60
		- fine color's ids under the color.
- all_category_name
	- np.array, h5py.h5r.Reference, (1, 23)
	- Each reference correspods the above cateogory keys under #refs#.
- all_color_name 
	- np.array, h5py.h5r.Reference, (1, 13)
	- Each reference correspods the above color keys under #refs#.


## Dataset Problem

- Mentioned in this paper's Figure 7, [(P. Tangseng, Z. Wu, & K. Yamaguchi. 2017) Looking at Outfit to Parse Clothing.](https://arxiv.org/pdf/1703.01386.pdf).
- [This repository](https://github.com/hrsma2i/fashion-parsing/tree/master/data/tmm_dataset_sharing) is the code of the above paper. This code deals with CFPD.
