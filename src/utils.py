import numpy as np


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]
