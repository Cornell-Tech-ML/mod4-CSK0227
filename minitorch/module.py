from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        """Initializes a new instance of the neural network module.

        This constructor sets up the basic structure for a neural network module:

        Attributes
        ----------
            _modules (dict): A dictionary to store sub-modules of this module.
            _parameters (dict): A dictionary to store the parameters of this module.
            training (bool): A flag indicating whether the module is in training mode.
                         Initialized to True by default.

        """
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the `training` flag of this and descendent to true."""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """Set the `training` flag of this and descendent to false."""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.

        """
        parameters: list[Tuple[str, Parameter]] = []

        for name, para in self._parameters.items():
            parameters.append((name, para))

        for module_name, module in self._modules.items():
            sub_params = module.named_parameters()
            parameters.extend(
                (f"{module_name}.{sub_name}", sub_param)
                for sub_name, sub_param in sub_params
            )

        return parameters

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        """Custom attribute setter for Module instances.

        This method intercepts attribute assignments and handles them based on the type of value:
        - If the value is a Parameter, it's stored in the _parameters dictionary.
        - If the value is another Module, it's stored in the _modules dictionary.
        - For all other types, it falls back to the default attribute setting behavior.

        Args:
        ----
            key (str): The name of the attribute being set.
            val (Parameter): The value to assign to the attribute.

        Returns:
        -------
            None

        """
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Custom attribute setter for Module instances.

        Args:
        ----
            key (str): The name of the attribute being set.
            val (Parameter): The value to assign to the attribute.

        Returns:
        -------
            None

        """
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the module callable.

        Args:
        ----
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
        -------
            Any: The output of the module's `forward` method.


        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initialize a Parameter object.

        Args:
        ----
            x (Any): The value to be stored in the Parameter. This can be of any type,
                    but is typically a tensor or array-like object.
            name (Optional[str]): An optional name for the Parameter. If provided and
                              the value supports naming, this name will be assigned
                              to the value. Defaults to None.

        Returns:
        -------
            None

        """
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
