# Sequential

"Sequential" is a common pattern in neural network frameworks, indicating a sequence of layers applied in order.

These are useful when building fairly straightforward models. But for anything nontrivial, subclass [`equinox.Module`][] instead.

---

::: equinox.nn.Sequential
    selection:
        members:
            - __init__
            - __call__

---

::: equinox.nn.Lambda
    selection:
        members:
            - __init__
            - __call__

---

::: equinox.nn.StatefulLayer
    selection:
        members:
            - __call__
