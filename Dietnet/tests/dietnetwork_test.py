import unittest
import os
import yaml
import sklearn
import numpy as np
import pandas as pd


class TestDietNetwork(unittest.TestCase):

    def setUp(self):
        # Note: using pre-made raw_data since compute node cannot connect to internet!
        if os.uname()[1] == 'kepler':
            config_file_path = 'tests/config/MNIST_tSNE_debug_kepler.yaml'
        else:
            config_file_path = 'tests/config/MNIST_tSNE_debug.yaml'

        print('loading: {}'.format(config_file_path))

        with open(config_file_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        _, _, _, _, builder = helpers.get_dset_info(self.config['data_config'], return_builder=True)

        data_path = os.path.join(self.config['data_config']['pca_config']['pca_data_dir'],
                                 self.config['data_config']['pca_config']['pca_data_name'] + '.pickle')

        _, algo_builder = helpers.run_algo(self.config['algo_config'],
                                           data_path=data_path,
                                           return_builder=True)

        self.builder = builder
        self.algo_builder = algo_builder

    def test_dloader_plus_algo(self):

        tracemalloc.start()

        raw_data = self.builder.get_raw_data()
        proc_data = self.builder.get_proc_data()
        pca_data, pca_obj = self.builder.get_pca_data()

        self.assertTrue(isinstance(raw_data, dict))
        self.assertTrue(isinstance(raw_data['data'], np.ndarray))
        self.assertTrue(isinstance(raw_data['labels'], np.ndarray))
        self.assertTrue(isinstance(proc_data['data'], np.ndarray))
        self.assertTrue(isinstance(pca_data, np.ndarray))
        self.assertTrue(isinstance(pca_obj, (sklearn.decomposition.PCA, sklearn.decomposition.IncrementalPCA)))

        # Get snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)

        pca_data, _ = self.builder.get_pca_data()
        embedding = self.algo_builder.get_embedding()

        self.assertTrue(embedding.shape[0] == pca_data.shape[0]), 'Number of samples changed after doing Dim. Res.!'
        self.assertTrue(isinstance(embedding, np.ndarray))

    def tearDown(self):
        # remove files
        os.remove(self.builder.proc_data_file)
        print('deleted {}'.format(self.builder.proc_data_file))
        os.remove(self.builder.pca_obj_file)
        print('deleted {}'.format(self.builder.pca_obj_file))
        os.remove(self.builder.pca_data_file)
        print('deleted {}'.format(self.builder.pca_data_file))

        try:
            os.remove(self.algo_builder.init_embed_file)
            print('deleted {}'.format(self.algo_builder.init_embed_file))
        except FileNotFoundError:
            pass
        os.remove(self.algo_builder.affinities_file)
        print('deleted {}'.format(self.algo_builder.affinities_file))

        for f in self.algo_builder.optim_files:
            try:
                os.remove(f)
                print('deleted {}'.format(f))
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    unittest.main()
