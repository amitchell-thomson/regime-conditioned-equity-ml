import pytest
from regime_ml.features.common.transform_parser import TransformParser, parse_transforms

def test_parse_chain_string():
    parser = TransformParser()
    chain = parser.parse_chain([{"diff": {"periods": 21}}, {"z_score": {"window": 252}}])
    assert len(chain.transforms) == 2
    assert chain.transforms[0].params == {"periods": 21}
    assert chain.transforms[1].params == {"window": 252}

def test_parse_chain_invalid_step():
    parser = TransformParser()
    with pytest.raises(ValueError):
        parser.parse_chain([{"diff": {"periods": 21}}, {"extra_key": {"InvalidKey": "InvalidValue"}}])

def test_get_feature_names():
    parser = TransformParser()
    chain = parser.parse_chain([{"diff": {"periods": 21}}, {"z_score": {"window": 252}}])
    feature_names = parser.get_feature_names("vix", [chain])
    assert len(feature_names) == 2
    assert feature_names[0] == "diff_21"
    assert feature_names[1] == "z_score_252"