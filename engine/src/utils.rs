pub mod constants {
    use nalgebra::Matrix4;
    // Flip the X axis so that +x points right
    pub const MATRIX_COORDINATE_X_FLIP: Matrix4<f32> = Matrix4::new(
        -1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}