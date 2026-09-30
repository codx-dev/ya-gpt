pub(crate) fn positive_usize(value: &str) -> Result<usize, String> {
    let value = value.parse::<usize>().map_err(|error| error.to_string())?;
    if value == 0 {
        return Err("must be greater than zero".into());
    }
    Ok(value)
}

pub(crate) fn unit_interval(value: &str) -> Result<f64, String> {
    let value = value.parse::<f64>().map_err(|error| error.to_string())?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("must be a finite number between 0.0 and 1.0 (inclusive)".into());
    }
    Ok(value)
}
