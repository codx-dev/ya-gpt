pub mod naive;

#[cfg(test)]
pub mod tests;

pub trait Engine {
    type Buffer;

    fn slice(&self, buffer: &Self::Buffer, offset: usize, len: usize) -> Self::Buffer;
    fn copy(&self, dst: &mut Self::Buffer, src: &Self::Buffer, ofs: usize);
    fn clone(&self, buffer: &Self::Buffer) -> anyhow::Result<Self::Buffer>;
    fn zeroes(&self, len: usize) -> anyhow::Result<Self::Buffer>;
    fn filled(&self, value: f64, len: usize) -> anyhow::Result<Self::Buffer>;
    fn buffer_len(&self, buffer: &Self::Buffer) -> usize;
    fn buffer_from_slice<S: AsRef<[f64]>>(&self, slice: S) -> anyhow::Result<Self::Buffer>;
    fn buffer_to_vec(&self, buffer: &Self::Buffer) -> anyhow::Result<Vec<f64>>;

    fn buffer_all_finite(&self, buffer: &Self::Buffer) -> anyhow::Result<bool>;

    #[allow(clippy::too_many_arguments)]
    fn adamw_in_place(
        &self,
        values: &mut Self::Buffer,
        gradients: &Self::Buffer,
        first_moment: &mut Self::Buffer,
        second_moment: &mut Self::Buffer,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        step: i32,
    ) -> anyhow::Result<()>;

    fn add(&self, a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer)
    -> anyhow::Result<()>;

    fn add_in_place(&self, out: &mut Self::Buffer, b: &Self::Buffer) -> anyhow::Result<()>;

    fn add_bias_in_place(
        &self,
        out: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()>;

    fn causal_mask_in_place(&self, out: &mut Self::Buffer, time: usize) -> anyhow::Result<()>;

    fn causal_mask_backward_in_place(
        &self,
        out: &mut Self::Buffer,
        time: usize,
    ) -> anyhow::Result<()>;

    fn concat_columns(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn cross_entropy(
        &self,
        logits: &Self::Buffer,
        targets: &[usize],
        rows: usize,
        vocab: usize,
    ) -> anyhow::Result<f64>;

    fn cross_entropy_backward(
        &self,
        logits: &Self::Buffer,
        targets: &[usize],
        rows: usize,
        vocab: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn dropout(
        &self,
        input: &Self::Buffer,
        mask: &Self::Buffer,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn dropout_in_place(&self, out: &mut Self::Buffer, mask: &Self::Buffer) -> anyhow::Result<()>;

    fn embedding(
        &self,
        table: &Self::Buffer,
        tokens: &[usize],
        vocab: usize,
        channels: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn embedding_backward(
        &self,
        tokens: &[usize],
        d_output: &Self::Buffer,
        vocab: usize,
        channels: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        input: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        rows: usize,
        cols: usize,
        epsilon: f64,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn layer_norm_backward(
        &self,
        input: &Self::Buffer,
        gamma: &Self::Buffer,
        d_output: &Self::Buffer,
        rows: usize,
        cols: usize,
        epsilon: f64,
        d_input: &mut Self::Buffer,
        d_gamma: &mut Self::Buffer,
        d_beta: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn matmul(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        rows: usize,
        inner: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn matmul_backward(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        d_output: &Self::Buffer,
        rows: usize,
        inner: usize,
        cols: usize,
        d_a: &mut Self::Buffer,
        d_b: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn relu(&self, input: &Self::Buffer, out: &mut Self::Buffer) -> anyhow::Result<()>;

    fn relu_backward(
        &self,
        input: &Self::Buffer,
        d_output: &Self::Buffer,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn scale(
        &self,
        input: &Self::Buffer,
        factor: f64,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn scale_in_place(&self, out: &mut Self::Buffer, factor: f64) -> anyhow::Result<()>;

    fn slice_columns(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        start: usize,
        width: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn softmax(&self, input: &Self::Buffer, out: &mut Self::Buffer) -> anyhow::Result<()>;

    fn softmax_in_place(&self, out: &mut Self::Buffer) -> anyhow::Result<()>;

    fn softmax_rows(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn softmax_rows_in_place(
        &self,
        out: &mut Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()>;

    fn softmax_rows_backward(
        &self,
        probabilities: &Self::Buffer,
        d_output: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn sum_rows(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn transpose(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn transpose_in_place(
        &self,
        out: &mut Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()>;
}
