import unittest
import ray

from rlskyjo.models.train_model_simple_rllib import (
    manual_training_loop,
    tune_training_loop,
)


class RayClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start it once for the entire test suite/module
        ray.init(local_mode=True)

    def test_tune_training_loop(self):
        tune_training_loop(timesteps_total=8000)

    def test_manual_training_loop(self):
        manual_training_loop(timesteps_total=8000)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()
