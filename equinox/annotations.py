import jax
import weakref

# So this is a bit of black magic.
#
# The things we want to annotate most often are JAX arrays, specifically jaxlib.xla_extension.DeviceArray.
# These don't support any kind of assignment to them, i.e. you can't do
# x = jnp.array([1])
# x.asdf = True
# moreover none of their attributes look amenable to sneaking something like this on to them.
#
# We can't use a global dictionary to record annotations, because JAX arrays aren't hashable.
# We can't use an id-dictionary to record annotations, because the same id might get reused later for a different
# object after this one has been deleted.
#
# We are saved by the fact that JAX arrays allow weak references, which produces the implementation below.


class _WeakIdDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_gets = 0

    def __setitem__(self, key, value):
        super().__setitem__(id(key), (weakref.ref(key), value))

    def __getitem__(self, key):
        weakref, value = super().__getitem__(id(key))
        if weakref() is None:
            # key happens to have the same `id` as a previous object that has since been deleted.
            del self[key]
            raise KeyError("Key not in dictionary.")
        self.num_gets += 1
        # Simple heuristic to clean out old data. Probably there's better ways of doing it by comparing
        # number of gets to length or number of sets and figuring out how densely the entires are being queried.
        if self.num_gets >= 1000:
            self.num_gets = 0
            for id_key, (weakref, value) in super().items():
                if weakref() is None:
                    super().__delitem__(id_key)
        return value

    def __delitem__(self, item):
        del self[id(item)]


_annotations = _WeakIdDict()

_single_leaf_treedef = jax.tree_structure(True)


def set_annotation(obj, annotation):
    # Not necessary for technical reasons. However only PyTree leaves will be iterated over and have the filter
    # functions applied, so this prevents an easy class of error.
    if jax.tree_structure(obj) != _single_leaf_treedef:
        raise ValueError("Only PyTree leaves can be annotated.")
    _annotations[obj] = annotation


_sentinel = object()


def get_annotation(obj, default=_sentinel):
    try:
        return _annotations[obj]
    except KeyError:
        if default is _sentinel:
            raise
        else:
            return _sentinel


def del_annotation(obj):
    del _annotations[obj]
