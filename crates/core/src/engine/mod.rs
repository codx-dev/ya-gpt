pub mod gpt;
pub mod naive;
pub mod optmizer;
pub mod types;

#[cfg(test)]
pub mod tests;

pub trait Engine {
    fn add(&self, a: &[f64], b: &[f64]) -> Vec<f64>;
    fn add_bias(&self, input: &[f64], bias: &[f64], rows: usize, cols: usize) -> Vec<f64>;
    fn causal_mask(&self, input: &[f64], time: usize) -> Vec<f64>;
    fn causal_mask_backward(&self, d_output: &[f64], time: usize) -> Vec<f64>;
    fn concat_columns(
        &self,
        a: &[f64],
        b: &[f64],
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Vec<f64>;
    fn cross_entropy(&self, logits: &[f64], targets: &[usize], rows: usize, vocab: usize) -> f64;
    fn cross_entropy_backward(
        &self,
        logits: &[f64],
        targets: &[usize],
        rows: usize,
        vocab: usize,
    ) -> Vec<f64>;
    fn dropout(&self, input: &[f64], mask: &[f64]) -> Vec<f64>;
    fn embedding(&self, table: &[f64], tokens: &[usize], vocab: usize, channels: usize)
    -> Vec<f64>;
    fn embedding_backward(
        &self,
        tokens: &[usize],
        d_output: &[f64],
        vocab: usize,
        channels: usize,
    ) -> Vec<f64>;
    fn layer_norm(
        &self,
        input: &[f64],
        gamma: &[f64],
        beta: &[f64],
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) -> Vec<f64>;
    fn layer_norm_backward(
        &self,
        input: &[f64],
        gamma: &[f64],
        d_output: &[f64],
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>);
    fn matmul(&self, a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64>;
    fn matmul_backward(
        &self,
        a: &[f64],
        b: &[f64],
        d_output: &[f64],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> (Vec<f64>, Vec<f64>);
    fn relu(&self, input: &[f64]) -> Vec<f64>;
    fn relu_backward(&self, input: &[f64], d_output: &[f64]) -> Vec<f64>;
    fn scale(&self, input: &[f64], factor: f64) -> Vec<f64>;
    fn slice_columns(
        &self,
        input: &[f64],
        rows: usize,
        cols: usize,
        start: usize,
        width: usize,
    ) -> Vec<f64>;
    fn softmax(&self, input: &[f64]) -> Vec<f64>;
    fn softmax_rows(&self, input: &[f64], rows: usize, cols: usize) -> Vec<f64>;
    fn softmax_rows_backward(
        &self,
        probabilities: &[f64],
        d_output: &[f64],
        rows: usize,
        cols: usize,
    ) -> Vec<f64>;
    fn sum_rows(&self, input: &[f64], rows: usize, cols: usize) -> Vec<f64>;
    fn transpose(&self, input: &[f64], rows: usize, cols: usize) -> Vec<f64>;
}
