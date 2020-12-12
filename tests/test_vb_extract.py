import numpy as np
import pytest

import vb_extract


class Test_parse_bracketed_param_name:
    """Test function parse_bracketed_param_name"""

    @pytest.mark.parametrize("param_name,expected", [
        ("alpha", ("alpha", tuple())),
        ("alpha_prime", ("alpha_prime", tuple())),
        ("beta[2]", ("beta", (2,))),
        ("beta[4,3,2]", ("beta", (4, 3, 2))),
        ("beta_prime[4,3,2]", ("beta_prime", (4, 3, 2))),
    ])
    def test_parse_bracketed_param_name(self, param_name, expected):
        """Test valid values"""
        assert expected == vb_extract.parse_bracketed_param_name(param_name)

    @pytest.mark.parametrize("param_name", ["a-b", "$a", "*", "", "ae[]"])
    def test_raise_value_error(self, param_name):
        """Test raise ValueError for invalid param_names"""
        with pytest.raises(ValueError):
            vb_extract.parse_bracketed_param_name(param_name)


class Test_extract_vb_samples:

    @pytest.mark.parametrize("stan_fit,expected", [
        ({
            "sampler_param_names": [],
            "sampler_params": []
        }, {}),
        ({
            "sampler_param_names": ["alpha"],
            "sampler_params": [[0]]
        }, {
            "alpha": np.array([0])
        }),
        ({
            "sampler_param_names": ["a[2]", "a[1]"],
            "sampler_params": [[1, 3, 5], [0, 2, 4]]
        }, {
            "a": np.arange(6).reshape((3, 2))
        }),
        ({
            "sampler_param_names": ["a[2,1]", "b", "a[1,1]"],
            "sampler_params": [[1, 3, 5], [6, 7, 8], [0, 2, 4]]
        }, {
            "a": np.arange(6).reshape((3, 2, 1)),
            "b": np.array([6, 7, 8])
        }),
        ({
            "sampler_param_names": ["a[2,1]", "b", "a[1,1]"],
            "sampler_params": [[], [], []]
        }, {
            "a": np.empty((0, 2, 1)),
            "b": np.empty((0,)),
        }),
    ])
    def test_extract_vb_samples(self, stan_fit, expected):
        np.testing.assert_equal(expected,
                                vb_extract.extract_vb_samples(stan_fit))

    @pytest.mark.parametrize("stan_fit", [
        {
            "sampler_param_names": ["a", "b"],
            "sampler_params": [[0], [0, 1]],
        },
        {
            "sampler_param_names": ["a[2,1]", "b", "a[1,1]"],
            "sampler_params": [[1], [6, 8], [2, 4]]
        },
        {
            "sampler_param_names": ["a[2,1]", "b", "a[1,1]"],
            "sampler_params": [[1, 2], [6, 8], [2, 4, 7]]
        },
        {
            "sampler_param_names": ["a[2,1]", "b", "a[1,1]"],
            "sampler_params": [[1, 3, 5], [6, 8], [0, 2, 4]]
        },
    ])
    def test_raises_value_error_invalid_sample_num(self, stan_fit):
        with pytest.raises(ValueError):
            vb_extract.extract_vb_samples(stan_fit)
