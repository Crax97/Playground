/*

let mut scene = Scene::new();

let mesh_1 = create_mesh(...);
let mesh_2 = create_mesh(...);

let material = create_material();

struct ScenePrimitive {
    mesh: Handle<Mesh>,
    material: Handle<Material>
    transform,
}

let mesh_handle_1 = scene.add(scene_1_mesh_primitive);
let mesh_handle_2 = scene.add(scene_2_mesh_primitive);

scene.edit(&mesh_handle_1).transform = make_new_transform();

let framebuffer = get_framebuffer();
{
    let renderer = make_renderer(&scene);
    renderer.render(&framebuffer);
}

swapchain.present();

*/
