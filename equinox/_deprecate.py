def deprecated_0_10(kwargs, name):
    try:
        kwargs[name]
    except KeyError:
        pass
    else:
        raise ValueError(
            f"{name} is deprecated as of Equinox version 0.10.0. Please see the "
            "release notes on GitHub for details on how to upgrade:\n"
            "https://github.com/patrick-kidger/equinox/releases/tag/v0.10.0"
        )
