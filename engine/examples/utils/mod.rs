use engine::{app::AppContext, scene_renderer::PointOfView};
use glam::{Quat, Vec3};
use winit::dpi::PhysicalPosition;

pub fn update_fps_camera(context: &AppContext, pov: &mut PointOfView) {
    const MOVEMENT_SPEED: f64 = 100.0;
    const ROTATION_DEGREES: f64 = 90.0;

    let mut camera_input = Vec3::default();

    if context.input.is_key_pressed(engine::input::Key::A) {
        camera_input.x = 1.0;
    } else if context.input.is_key_pressed(engine::input::Key::D) {
        camera_input.x = -1.0;
    }

    if context.input.is_key_pressed(engine::input::Key::W) {
        camera_input.z = 1.0;
    } else if context.input.is_key_pressed(engine::input::Key::S) {
        camera_input.z = -1.0;
    }
    if context.input.is_key_pressed(engine::input::Key::Q) {
        camera_input.y = 1.0;
    } else if context.input.is_key_pressed(engine::input::Key::E) {
        camera_input.y = -1.0;
    }

    camera_input *= (MOVEMENT_SPEED * context.time.delta_seconds()) as f32;

    let mouse_delta = context.input.normalized_mouse_position();
    let pov_transform = pov.transform;

    let (cam_pitch, cam_roll, _) = pov_transform.rotation.to_euler(glam::EulerRot::XYZ);
    let mut cam_roll = cam_roll.to_degrees();
    let mut cam_pitch = cam_pitch.to_degrees();

    cam_roll -= mouse_delta.x * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
    cam_pitch += mouse_delta.y * (ROTATION_DEGREES * context.time.delta_seconds()) as f32;
    cam_pitch = cam_pitch.clamp(-89.0, 89.0);

    let new_location_offset = camera_input.x * pov.transform.left()
        + camera_input.y * pov.transform.up()
        + camera_input.z * pov.transform.forward();
    pov.transform.location += new_location_offset;
    pov.transform.rotation = Quat::from_euler(
        glam::EulerRot::XYZ,
        cam_pitch.to_radians(),
        cam_roll.to_radians(),
        0.0,
    );

    let cursor_position = context.window().inner_size();
    context
        .window()
        .set_cursor_position(PhysicalPosition::new(
            cursor_position.width / 2,
            cursor_position.height / 2,
        ))
        .unwrap();
}
