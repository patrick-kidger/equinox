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
    def __setitem__(self, key, value):
        super().__setitem__(id(key), (weakref.ref(key), value))

    def __getitem__(self, key):
        weakref, value = super().__getitem__(id(key))
        if weakref() is None:
            # key happens to have the same `id` as a previous object that has since been deleted.
            del self[key]
            raise KeyError("Key not in dictionary.")
        return value

    def __delitem__(self, item):
        del self[id(item)]


_annotations = _WeakIdDict()


def set_annotation(obj, annotation):
    _annotations[obj] = annotation


def get_annotation(obj):
    return _annotations[obj]


def del_annotation(obj):
    del _annotations[obj]
