# Echo Luna Rust Extensions

This crate provides optional Rust-powered acceleration for computationally intensive
sections of the Echo-Luna project. The library is compiled as a Python extension
module using [PyO3](https://pyo3.rs/) and distributed via `maturin`.

## Building

To build the extension in editable mode run:

```bash
maturin develop
```

The generated module exposes helpers that speed up vector serialisation for the
`VectorizedImage` type. When the compiled module is unavailable the Python
implementation remains as a compatible fallback.
