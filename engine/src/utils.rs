#[rustfmt::skip]
pub(crate) mod constants {
    use nalgebra::Matrix4;
    pub(crate) const Z_INVERT_MATRIX: Matrix4<f32> = 

        Matrix4::<f32>::new(
        1.0, 0.0, 0.0, 0.0, 
        0.0, -1.0, 0.0, 0.0, 
        0.0, 0.0, -1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0,
    );
}
