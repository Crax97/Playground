use std::collections::HashMap;

use mgpu::ShaderModule;

use crate::immutable_string::ImmutableString;

#[derive(Default)]
pub struct ShaderCache {
    shaders: HashMap<ImmutableString, ShaderModule>,
}

impl ShaderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_shader(
        &mut self,
        identifier: impl Into<ImmutableString>,
        shader_module: ShaderModule,
    ) {
        let old = self.shaders.insert(identifier.into(), shader_module);
        if old.is_some() {
            panic!("A shader with this identifier was already defined");
        }
    }

    pub fn get_shader_module(&self, identifier: &ImmutableString) -> ShaderModule {
        self.shaders
            .get(identifier)
            .copied()
            .expect("No shader found with this identifier")
    }
}
