
# Hide the warning messages about CPU/GPU
import TensorflowUtils as utils
import glob
from tensorflow.python.platform import gfile
from six.moves import cPickle as pickle
import random
import numpy as np
from tqdm import tqdm
import os
import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from FCN, which trained on PASCAL VOC 2011
image_mean = [104.00699, 116.66877, 122.67892]
# image_mean = [0.6073780437240768, 0.5761329233683282, 0.5329315454579794]; # from Calculation


def read_dataset(data_dir):
    # sample record: {'image': f, 'annotation': annotation_file,
    # 'filename': filename}
    training_records = []
    validation_records = []
    testing_records = []
    all_records = []

    data_dir = "E:/Dataset/CFPD/"
    image_dir = data_dir + "image/"
    annotation_file_path = data_dir + "fashion_parsing_data.mat"

    print("## Image dir:", image_dir)
    image_list = os.listdir(image_dir)
    # fashion_dataset = read_mat(annotation_file_path)
	fashion_dataset = convert_mat_to_dict(annotation_file_path)

    for i, each in enumerate(image_list):
        filename = image_dir + each
        record = {'image': None, 'annotation': None, 'filename': None,
                  'category_label': None, 'color_label': None, 'img_name': None}
        record['image'] = filename
        record['filename'] = filename
		
        # record['category_label'] = fashion_dataset[i][0]
        # record['color_label'] = fashion_dataset[i][1]
        # record['img_name'] = fashion_dataset[i][2]
        # record['annotation'] = fashion_dataset[i][3]
		
        record['category_label'] = fashion_dataset[i]['category_label']
        record['color_label'] = fashion_dataset[i]['color_label']
        record['img_name'] = fashion_dataset[i]['img_name']
        record['annotation'] = fashion_dataset[i]['segmentation']
		
        all_records.append(record)

    np.random.shuffle(all_records)
    num_records = len(all_records)
    training_records = all_records[0:int(num_records*0.78)]
    validation_records = all_records[int(num_records*0.78):int(num_records*0.8)]
    testing_records = all_records[int(num_records*0.8):num_records]

    return training_records, validation_records, testing_records


def read_mat(annotation_file_path):
    fashion_dataset = []

    with h5py.File(annotation_file_path, 'r') as file:
        print(list(file.keys()))
        # print(file)
        # ['#refs#', 'all_category_name', 'all_colors_name', 'fashion_dataset']
        fashion_data = file['fashion_dataset']
        # ['category_label', 'color_label', 'img_name', 'segmentation']

        for each in tqdm(fashion_data):
            temp = []
            temp.append(hdf5_to_list(file[each[0]]['category_label']))
            temp.append(hdf5_to_list(file[each[0]]['color_label']))
            temp.append(hdf5_to_list(file[each[0]]['img_name']))
            temp.append(hdf5_to_list(file[each[0]]['segmentation']))
            fashion_dataset.append(temp)

    return fashion_dataset
    
    
def hdf5_to_list(data):
    x = data[:]
    #x = x.tolist()
    return x

	
def convert_mat_to_dict(mat_file='fashion_parsing_data.mat'):
    f = h5py.File(mat_file, 'r')
    all_ctgs = get_all_ctgs(f)
    iter_ = iter(f.get('#refs#').values())
	
    df = pd.DataFrame()
    for outfit in tqdm(iter_, total=len(f.get('#refs#'))):
        try:
            # img_name
            ascii_codes = list(outfit.get('img_name').value[:,0])
            img_name = ''.join([chr(code) for code in ascii_codes ])
            print(img_name)
            
            # super pix 2 category
            spix2ctg = outfit.get('category_label').value[0]
            #pd.Series(spix2ctg).value_counts().plot(kind='bar')
            #print(spix2ctg.shape)
            #plt.plot(spix2ctg)
            #plt.show()
            
            # super pix 2 color
            spix2clr = outfit.get('color_label').value[0]
            #print(spix2clr.shape)
            #plt.plot(spix2clr)
            #plt.show()

            # super pix
            spixseg = outfit.get('segmentation').value.T
            #print(spixseg.shape)
            # plt.imshow(spixseg)
            #plt.plot(spixseg)
            #plt.show()
            # plt.savefig('image.png')

            # super pix -> semantic segmentation
            semseg = np.zeros(spixseg.shape)
            for i, c in enumerate(spix2ctg):
                semseg[spixseg == i] = c-1

            # semseg -> bbox
            items = []
            for i, ctg in enumerate(all_ctgs):
                region = np.argwhere(semseg == i)
                if region.size != 0:
                    bbox = {
                        'ymin':int(region.min(0)[0]),
                        'xmin':int(region.min(0)[1]),
                        'ymax':int(region.max(0)[0]),
                        'xmax':int(region.max(0)[1]),
                    }
                    items.append({
                        'bbox': bbox,
                        'category': ctg,
                    })

            df = df.append({
                'img_name': img_name,
                'category_label': category_label,
                'color_label': color_label,
                'segmentation': segmentation,
                'items': items,
            }, ignore_index=True)
        except AttributeError:
            pass

    d = df.to_dict(orient='records')
    return d

	
"""
    create image filename list for training and validation data (input and annotation)
"""


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]  # Linux
                # filename = os.path.splitext(f.split("\\")[-1])[0]  # windows
                annotation_file = os.path.join(
                    image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {
                        'image': f,
                        'annotation': annotation_file,
                        'filename': filename}
                    image_list[directory].append(record)
                else:
                    print(
                        "Annotation file not found for %s - Skipping: %s" %
                        (filename, annotation_file))

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, no_of_images))

    return image_list
