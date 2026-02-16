use pyo3::prelude::*;
use polars::prelude::*;
use numpy::{IntoPyArray, PyArray1};
use std::collections::VecDeque;

#[pyfunction]
fn rolling_std_brightness(py: Python, brightness: Vec<f64>, window: usize) -> Py<PyArray1<f64>> {
    let mut result = Vec::with_capacity(brightness.len());
    let mut window_vals = VecDeque::new();

    for &val in &brightness {
        window_vals.push_back(val);
        if window_vals.len() > window {
            window_vals.pop_front();
        }
        let mean = window_vals.iter().sum::<f64>() / window_vals.len() as f64;
        let std = (window_vals.iter().map(|v| (*v - mean).powi(2)).sum::<f64>() / window_vals.len() as f64).sqrt();
        result.push(std);
    }

    result.into_pyarray(py).to_owned()
}

#[pymodule]
fn astro_plugin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_std_brightness, m)?)?;
    Ok(())
}
