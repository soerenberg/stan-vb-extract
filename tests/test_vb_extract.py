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
