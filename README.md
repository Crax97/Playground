
This repo contains my "Playground", a set of tools and libraries i use to study various computer-graphics related topics.
At the moment, the project is structured in separate crates, with the most important being:
* `gpu`, a graphics abstraction library: it started as a way to learn Vulkan, and i'm slowly transitioning it to be a generic API with the idea of supporting
DX12 and OpenGL as well in the future;
* `engine`, which contains a scene abstraction, a simple render graph and a deferred renderer, built using `gpu`;
* `testbench`, which contains various binaries that i use to test my work: at the moment i have
    * `compute`, a simple executable that showcases `gpu`'s compute shader support;
    * `planes`, which renders three textured planes on screen
    * `gltf_viewer`, a gltf loader and renderer

### Disclaimer
This project has only been tested on a Linux laptop with a NVidia RTX 3070 and a Windows computer with an AMD 5700XT: 
Given the intrinsic complexity of Vulkan and the modern graphics API, i don't guarantee that it will work on your machine.
