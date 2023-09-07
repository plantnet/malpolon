import numpy as np
import timm

from malpolon.models.standard_prediction_systems import GenericPredictionSystem


def test_state_dict_replace_key():
    model = timm.create_model('resnet18')
    sd = model.state_dict()
    keys = np.array(list(sd.keys()))
    keys_pos = np.where(np.char.find(keys, 'fc') != -1)[0]

    sd_replace_key = ['fc', 'layer5']
    sd_new = GenericPredictionSystem.state_dict_replace_key(sd, sd_replace_key)
    keys_new = np.array(list(sd_new.keys()))
    res = lambda keys, pos: all((sd_replace_key[1] in keys[i]) and
                                (sd_replace_key[0] not in keys[i]) for i in pos)
    assert res(keys_new, keys_pos)