import equinox as eqx
import gc
import jax.numpy as jnp
import pytest


def _assert_annotations_empty():
    assert len(eqx.annotations._annotations) == 0


def test_get_set_annotations():
    pass_objs = [1, jnp.array([2., 3.])]
    fail_objs = [set("a"), [3, {"a": jnp.array([4., 5.]), "b": None}, None]]
    for obj in pass_objs:
        eqx.set_annotation(obj, {"frozen": True})
    for obj in fail_objs:
        with pytest.raises(ValueError):
            eqx.set_annotation(obj, {"frozen": True})
    for obj in pass_objs:
        assert eqx.get_annotation(obj) == {"frozen": True}
    del pass_objs
    gc.collect()
    _assert_annotations_empty()


def test_del_annotations():
    eqx.set_annotation(2, True)
    with pytest.raises(KeyError):
        eqx.get_annotation(3)
    with pytest.raises(KeyError):
        eqx.del_annotation(3)
    eqx.del_annotation(2)
    _assert_annotations_empty()


def test_default_get_annotation():
    eqx.set_annotation(2, True)
    assert eqx.get_annotation(3, default="hi") == "hi"
    assert eqx.get_annotation(2) is True
    eqx.del_annotation(3)
    eqx.del_annotation(2)
    _assert_annotations_empty()
