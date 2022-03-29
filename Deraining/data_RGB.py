import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestRainL, DataLoaderTestRainH

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)

def get_test_rain_L_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestRainL(rgb_dir, img_options)

def get_test_rain_H_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestRainH(rgb_dir, img_options)
