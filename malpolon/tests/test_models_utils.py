"""This script tests the models.utils module.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

import unittest
from pathlib import Path, PosixPath
from typing import Mapping, Union

import timm
from hydra import compose, initialize
from omegaconf import DictConfig
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from malpolon.models.utils import check_optimizer, check_scheduler

CONFIGS_PATH = Path('data/test_models_utils_configs/')
MODEL = timm.create_model("resnet18", pretrained=True)
OPTIMIZER = optim.Adam(MODEL.parameters())

def load_config(config_name: Union[str, PosixPath]) -> DictConfig:
    # Initialize the Hydra environment
    with initialize(config_path=str(CONFIGS_PATH), version_base='1.3'):
        # Load the configuration file
        cfg = compose(config_name=config_name)
        return cfg

def test_check_optimizer() -> None:
    # 1 optimizer (instanciated), 0 scheduler
    res_opt, res_sch = check_optimizer(OPTIMIZER, MODEL)
    assert isinstance(res_opt, list) and isinstance(res_sch, list)
    assert res_sch == [None]
    assert isinstance(res_opt[0], Optimizer)

    # 1 optimizer (config), 0 scheduler
    opt_config = load_config('1_opt_0_sch.yaml').optimizer
    res_opt, res_sch = check_optimizer(opt_config['optimizer'], MODEL)
    assert isinstance(res_opt, list) and isinstance(res_sch, list)
    assert res_sch == [None]
    assert isinstance(res_opt[0], Optimizer)

    # 1 optimizer (config), 1 scheduler (config)
    opt_config = load_config('1_opt_1_sch.yaml').optimizer
    res_opt, res_sch = check_optimizer(opt_config['optimizer'], MODEL)
    assert isinstance(res_opt, list) and isinstance(res_sch, list)
    assert isinstance(res_opt[0], Optimizer)
    assert isinstance(res_sch[0], Mapping)
    assert isinstance(res_sch[0]['scheduler'], LRScheduler)

    # 2 optimizers (config), 1 scheduler (config)
    opt_config = load_config('2_opt_1_sch.yaml').optimizer
    res_opt, res_sch = check_optimizer(opt_config['optimizer'], MODEL)
    assert isinstance(res_opt, list) and isinstance(res_sch, list)
    assert len(res_opt) == 2 and len(res_sch) == 2
    assert all(isinstance(opt_i, Optimizer) for opt_i in res_opt)
    assert res_sch[0] is None
    assert isinstance(res_sch[1], Mapping)
    assert isinstance(res_sch[1]['scheduler'], LRScheduler)

    # 2 optimizers (config), 2 schedulers (config)
    opt_config = load_config('2_opt_2_sch.yaml').optimizer
    res_opt, res_sch = check_optimizer(opt_config['optimizer'], MODEL)
    assert isinstance(res_opt, list) and isinstance(res_sch, list)
    assert len(res_opt) == 2 and len(res_sch) == 2
    assert all(isinstance(opt_i, Optimizer) for opt_i in res_opt)
    assert all(isinstance(sch_i, Mapping) for sch_i in res_sch)
    assert all(isinstance(sch_i['scheduler'], LRScheduler) for sch_i in res_sch)

# Config file edge cases error testing
class TestCheckOptimizerEdgeCases(unittest.TestCase):    
    ## Error_1: 'optimizer' key content is None
    def test_error_1(self) -> None:
        opt_config = load_config('error_1.yaml').optimizer
        res_opt, res_sch = check_optimizer(opt_config['optimizer'], MODEL)
        assert isinstance(res_opt, list) and isinstance(res_sch, list)
        assert len(res_opt) == 0 and len(res_sch) == 0

    ## Error_2: 1st optimizer key content is None
    def test_error_2(self) -> None:
        opt_config = load_config('error_2.yaml').optimizer
        with self.assertRaises((TypeError, ValueError)):
            _, _ = check_optimizer(opt_config['optimizer'], MODEL)

    ## Error_3: scheduler key content is None
    def test_error_3(self) -> None:
        opt_config = load_config('error_3.yaml').optimizer
        res_opt, res_sch = check_optimizer(opt_config['optimizer'], MODEL)
        assert isinstance(res_opt, list) and isinstance(res_sch, list)
        assert len(res_opt) == 1 and len(res_sch) == 1
        assert res_sch[0] is None
    
    ## Error_4: lr_scheduler_config key is None
    ### This behavior can raise errors depending on the LRScheduler used, but said error is raised by PyTorchLightning's trainer when ran.
    ### Here, it is un-testable, unless testing an exhaustive list of LRSchedulers needing a lr_scheduler_config key.
    def test_error_4(self) -> None:
        pass

    ## Error_5: lr_scheduler_config key content is None or wrong
    ### Same idea, if there are missing or wrong keys, the error will not be raised at this stage.
    ### Plus, I do not know for sure the exhaustive list of valid keys for a lr_scheduler_config dict.
    def test_error_5(self) -> None:
        pass

    ## Error_6: an optimizer is passed with a custom name and no callable
    def test_error_6(self) -> None:
        opt_config = load_config('error_6.yaml').optimizer
        with self.assertRaises(KeyError):
            _, _ = check_optimizer(opt_config['optimizer'], MODEL)

    ## Error_7: a scheduler is passed with a custom name and no callable
    def test_error_7(self) -> None:
        opt_config = load_config('error_7.yaml').optimizer
        with self.assertRaises(KeyError):
            _, _ = check_optimizer(opt_config['optimizer'], MODEL)

    ## Error_8: an optimizer's kwargs are invalid
    def test_error_8(self) -> None:
        opt_config = load_config('error_8.yaml').optimizer
        with self.assertRaises(TypeError):
            _, _ = check_optimizer(opt_config['optimizer'], MODEL)

    ## Error_9: a scheduler's kwargs are invalid
    def test_error_9(self) -> None:
        opt_config = load_config('error_9.yaml').optimizer
        with self.assertRaises(TypeError):
            _, _ = check_optimizer(opt_config['optimizer'], MODEL)
    

def test_check_scheduler() -> None:
    sch = lr_scheduler.StepLR(OPTIMIZER, step_size=30, gamma=0.1)

    # None scheduler
    res = check_scheduler(None, OPTIMIZER)
    assert res is None

    # Instanciated scheduler
    res = check_scheduler(sch, OPTIMIZER)
    assert isinstance(res, Mapping)
    assert isinstance(res["scheduler"], LRScheduler)

    # Config scheduler
    sch_config = {
        "reduce_lr_on_plateau": {
            "kwargs": {"threshold": 0.001},
            "lr_scheduler_config": {
                "scheduler": "reduce_lr_on_plateau",
                "monitor": "loss/val",
            },
        }
    }
    res = check_scheduler(sch_config, OPTIMIZER)
    assert isinstance(res, Mapping)
    assert isinstance(
        res["scheduler"], LRScheduler
    )  # requires torch >= 2.2.0 to work with ReduceLROnPlateau as on anterior versions, ReduceLROnPlateau does not inherit LRScheduler
    assert isinstance(res["scheduler"], lr_scheduler.ReduceLROnPlateau)
