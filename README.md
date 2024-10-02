# Explore your FARGO3D simulation outputs with `Fargonaut`!

Fargonaut is a package for post-processing and visualising outputs of [FARGO3D](https://github.com/FARGO3D/fargo3d), written in Python.

[![Build and test](https://github.com/dc2917/Fargonaut/actions/workflows/ci.yml/badge.svg)](https://github.com/dc2917/Fargonaut/actions/workflows/ci.yml)


```py
>>> from fargonaut.output import Output

>>> output = Output("/path/to/fargo3d/outputs/fargo")
>>> gasdens50 = output.get_field("gasdens", 50)
>>> gasenergy50 = output.get_field("gasenergy", 50)
>>> gaspressure50 = gasdens50 * gasdenergy50**2
>>> gaspressure50.plot()
```

![Gas pressure output 50](docs/images/fargo_gaspressure50.png)

See the [documentation](https://dc2917.github.io/Fargonaut/index.html) for installation instructions, example usage and the API reference.

## Contributing to Fargonaut

Contributions to Fargonaut are welcome. Please see the [contributing guidelines](CONTRIBUTING.md).

## License

Fargonaut is fully open source. For more information about its license, see [LICENSE](LICENSE).
