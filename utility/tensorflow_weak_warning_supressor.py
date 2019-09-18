import warnings
import os


def execute():

    """
    Disable "FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated.
             In future, it will be treated as `np.float64 == np.dtype(float).type`."

    https://github.com/h5py/h5py/issues/995
    """
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message="Conversion of the second argument of issubdtype "
    #                                               "from `float` to `np.floating` is deprecated", category=FutureWarning)
    #     import h5py





    """
    Disable "I tensorflow/core/platform/cpu_feature_guard.cc:140] 
             Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA."
             
    https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
    
    You can use TF environment variable TF_CPP_MIN_LOG_LEVEL:
            - It defaults to 0, displaying all logs
            - To filter out INFO logs set it to 1
            - WARNINGS additionally, 2
            - and to additionally filter out ERROR logs set it to 3
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




