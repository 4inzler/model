use numpy::{IxDyn, PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::fmt::Write;

#[pyfunction]
fn serialize_vector(array: PyReadonlyArray1<f64>) -> PyResult<String> {
    let slice = array.as_slice()?;
    if slice.is_empty() {
        return Ok(String::new());
    }

    let mut output = String::with_capacity(slice.len() * 8);
    let mut iter = slice.iter();
    if let Some(first) = iter.next() {
        write!(&mut output, "{:.6}", first)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to format value: {err}")))?;
    }

    for value in iter {
        output.push(',');
        write!(&mut output, "{:.6}", value)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to format value: {err}")))?;
    }

    Ok(output)
}

#[pyfunction]
fn deserialize_vector(
    py: Python<'_>,
    data: &str,
    shape: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    if shape.is_empty() {
        return Err(PyValueError::new_err(
            "shape must contain at least one dimension",
        ));
    }

    let expected = shape.iter().product::<usize>();
    let mut values = Vec::with_capacity(expected);

    for token in data.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<f64>() {
            Ok(value) => values.push(value),
            Err(err) => {
                return Err(PyValueError::new_err(format!(
                    "failed to parse float '{trimmed}': {err}"
                )))
            }
        }
    }

    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "expected {expected} values, found {}",
            values.len()
        )));
    }

    let array = PyArrayDyn::<f64>::zeros(py, IxDyn(&shape), false);
    let mut slice = array.as_slice_mut()?;
    slice.copy_from_slice(&values);

    Ok(array.into())
}

#[pymodule]
fn echo_luna_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize_vector, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_vector, m)?)?;
    Ok(())
}
