"""Decorator to handle function serialization"""

import econpizza as ep
import os

try:
    from jax import export
except:
    export = None

import jax.numpy as jnp
import jax
import sys

def export_and_serialize(
    args,
    kwargs,
    func,
    func_name,
    shape_struct,
    vjp_order,
    skip_jitting,
    export_with_kwargs=False,
    reuse_first_item_scope=False
):
    """
    Export and serialize a function with given symbolic shapes.

    Args:
        func (function): The function to be exported and serialized. If `skip_jitting` is True, then `func` needs to be jitted.
        func_name (str): The name of the function to be used for the serialized file.
        shape_struct (dict): A dictionary defining the shape and type of the function's inputs.
        vjp_order (int): The order of the vector-Jacobian product.
        skip_jitting (bool): Whether to skip JIT compilation.

    Returns:
        function: The exported and serialized function ready to be called.
    """
    scope = export.SymbolicScope()
    poly_args = map_shape_struct_dict_to_jax_shape(shape_struct, scope)

    function_to_export = func if skip_jitting else jax.jit(func)

    if export_with_kwargs == True:
        poly_kwargs = _prepare_poly_kwargs(shape_struct, kwargs, poly_args)
        
        exported_func: export.Exported = export.export(function_to_export)(
            **poly_kwargs
        )
    else:
        exported_func: export.Exported = export.export(function_to_export)(*poly_args)

    # Save exported artifact
    serialized_path = os.path.join(ep.config.econpizza_cache_folder, f"{func_name}")
    serialized: bytearray = exported_func.serialize(vjp_order=vjp_order)
    
    with open(serialized_path, "wb") as file:
        file.write(serialized)

    
    return exported_func.call


def cacheable_function_with_export(
    func_name, shape_struct, alternative_shape_struct=None, vjp_order=0, skip_jitting=False, export_with_kwargs=False, reuse_first_item_scope=False
):
    """
    Decorator to replace function with exported and cached version if caching is enabled.

    Args:
        func_name (str): The name under which the function will be saved. "my_func" will be saved as "my_func.bin" on disk.
        shape_struct (dict): A dictionary defining the shape and type of the function's inputs.
        vjp_order (int, optional): The order of the vector-Jacobian product. Defaults to 0.
        skip_jitting (bool, optional): Whether to skip JIT compilation. Defaults to False.

    Returns:
        function: The decorated function which uses the cached version if available, otherwise the original function.

    Usage:
        @cacheable_function_with_export("f", {"x": ("a,", jnp.float64)})
        def f(x):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if ep.config.enable_persistent_cache == True:
                _check_jax_export_dependencies_and_raise()

                def _get_func_name(shape_mismatch: bool):
                    return f"{func_name}_alt" if shape_mismatch else func_name
                
                def _get_shape_struct(shape_mismatch: bool):
                    return alternative_shape_struct if shape_mismatch and alternative_shape_struct else shape_struct

                filtered_kwargs = {key: value for key, value in kwargs.items() if key in shape_struct}

                # First, check if the function is already serialized
                # If the function is serialized, deserialize it and return the .call
                # Else, export, serialize and return the call.
                serialized_path = os.path.join(
                    ep.config.econpizza_cache_folder, f"{func_name}"
                )
                serialized = _read_serialized_function(serialized_path)
                # Load alternative function serialized object
                serialized_alt_path = os.path.join(
                    ep.config.econpizza_cache_folder, f"{func_name}_alt"
                )
                serialized_alt = _read_serialized_function(serialized_alt_path)

                if serialized_alt:
                    cached_func = export.deserialize(serialized_alt)
                    args_shapes, exported_shapes = _get_args_exported_shapes(args, cached_func)
                    
                    # If the args and specs match, call alternative export
                    if _check_shape_dim_size(args_shapes, exported_shapes) == True:
                        return cached_func.call(*args, **filtered_kwargs)
                    else:
                        pass

                shape_mismatch = False
                if serialized and alternative_shape_struct is not None and not serialized_alt:
                    cached_func = export.deserialize(serialized)
                    # Check shapes of cached_func.in_avals and args
                    if not export_with_kwargs:
                        args_shapes, exported_shapes = _get_args_exported_shapes(args, cached_func)

                        # Dimensions mismatch
                        if _check_shape_dim_size(args_shapes, exported_shapes) == False:
                            serialized = None
                            shape_mismatch = True


                if serialized:
                    cached_func = export.deserialize(serialized)
                    return cached_func.call(*args, **filtered_kwargs)
                else:
                    cached_func = export_and_serialize(
                        args=args,
                        kwargs=kwargs,
                        func=func,
                        func_name=_get_func_name(shape_mismatch),
                        shape_struct=_get_shape_struct(shape_mismatch),
                        vjp_order=vjp_order,
                        skip_jitting=skip_jitting,
                        export_with_kwargs=export_with_kwargs,
                        reuse_first_item_scope=reuse_first_item_scope
                    )
                    return cached_func(*args, **filtered_kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def map_shape_struct_dict_to_jax_shape(node, scope):
    """Generate a jax.ShapeDTypeStruct from a dictionary of polymorphic shapes.

    Args:
        node (tuple | dict | list): an element from the shape (example: ("a", jnp.int64))
        scope (export.SymbolicScope): the scope which the symbolic shape should use. As this is constructing a whole
        structure with related objects, they should share the same scope

    Returns:
        jax.ShapeDtypeStruct: shape struct using symbolic shape
    """
    if (
        isinstance(node, tuple)
        and (len(node) == 2
        or len(node) == 3)
        and not isinstance(node[0], tuple)
    ):
        if len(node) == 3:
            value, dtype, constraint = node
        else:
            value, dtype = node
            constraint = None

        if constraint:
            shape_poly = export.symbolic_shape(shape_spec=value, constraints=constraint)
        else:
            shape_poly = export.symbolic_shape(shape_spec=value, scope=scope)

        if scope is None:
            return jax.ShapeDtypeStruct(shape_poly, dtype=dtype), shape_poly[0].scope
        else:
            return jax.ShapeDtypeStruct(shape_poly, dtype=dtype)
    elif isinstance(node, dict):
        return tuple(
            map_shape_struct_dict_to_jax_shape(v, scope) for v in node.values()
        )
    elif isinstance(node, (list, tuple)):
        return type(node)(map_shape_struct_dict_to_jax_shape(v, scope) for v in node)
    else:
        return node


def _read_serialized_function(serialized_path):
    if os.path.exists(serialized_path):
        with open(serialized_path, "rb") as file:
            serialized = file.read()
    else:
        serialized = None

    return serialized
                
def _prepare_poly_kwargs(shape_struct, kwargs, poly_args):
    func_kwargs_names = list(shape_struct.keys())
    poly_kwargs = {key: value for key, value in zip(func_kwargs_names, poly_args)}

    for key, value in kwargs.items():
        if key not in poly_kwargs:
            poly_kwargs[key] = value

    assert len(poly_kwargs) == len(kwargs), "Keyword argument missing in shape poly"

    return poly_kwargs


def _check_shape_dim_size(current_arg_shapes: list[tuple], exported_args_shapes: list[tuple]):
    for current_shape, exported_shape in zip(current_arg_shapes, exported_args_shapes):
        if len(current_shape) != len(exported_shape): return False
    
    return True

def _get_args_exported_shapes(args, exported):
    args_flatten, _ = jax.tree_util.tree_flatten(args)
                        
    args_shapes = [arg.shape if hasattr(arg, 'shape') else () for arg in args_flatten]
    exported_shapes = [exported.shape if hasattr(exported, 'shape') else () for exported in exported.in_avals]

    return args_shapes, exported_shapes

    
def _check_jax_export_dependencies_and_raise():
    try:
        import absl
    except ImportError as e:
        raise ImportError('Please install absl-py in order to use jax export')
    
    try:
        import flatbuffers
    except ImportError as e:
        raise ImportError('Please install flatbuffers in order to use jax export')

