
This repo contains my "Playground", a set of tools and libraries i use to study various computer-graphics related topics.
At the moment, the project is structured in separate crates, with the most important being:
* `mgpu`, a graphics abstraction library: it started as a way to learn Vulkan, and i'm slowly transitioning it to be a generic API with the idea of supporting
DX12 and OpenGL as well in the future.
`mgpu` is split in three main modules:
    * `hal` is a low-level graphics API abstraction (currently only Vulkan is implemented)
    * `rdg` is a rendergraph implementation built upon `hal`
    * `mgpu` (mainly implemented in `device.rs`) is the higher-level api meant for the user, built upon `hal` and `rdg`
    * `examples` contains some `mgpu` examples, which can be run with `cargo run --example example_name` (e.g `cargo run --example textured_cube`)
    
* `egui-mgpu`: an integration for the egui UI library using mgpu;
* `engine`, which contains a scene abstraction, a simple render graph and a deferred renderer, built using `mgpu`;
* `gltf_viewer`, a viewer for gltf files implementing the Khronos PBR core shading model using standard deferred rendering:
    * By pressing the F keys, various debug viewmodes can be selected:
        | Key   | Mode          |
        |-------|---------------|
        | F1    | BaseColor     |
        | F2    | Normals       |
        | F3    | EmissiveAO    |
        | F4    | WorldPosition |   
        | F9    | Final image   |   
    * To interact with the scene, the mouse must be captured with ;
    * the O key switches a panning camera
        *  Left mouse moves the camera, Right mouse zooms in/out
    * P switches to a free camera with wasd movement 

To run the gltf_viewer, use `cargo run --bin gltf_viewer -- --file path_to_gltf`, e.g to open the WaterBottle sample execute the following command (make sure the repository submodules were cloned):
`cargo run --bin gltf_viewer -- --file .\glTF-Sample-Assets\Models\WaterBottle\glTF\WaterBottle.gltf`

### Disclaimer
This project has only been tested on a Linux laptop with a NVidia RTX 3070 and a Windows computer with an AMD 7800XT: 
Given the intrinsic complexity of Vulkan and the modern graphics API, i don't guarantee that it will work on your machine.
