use std::collections::hash_map::Entry;
use std::collections::HashMap;

use gpu::UniformVariableDescription;
use gpu::{
    render_pass_2::RenderPass2, Binding2, BufferCreateInfo, BufferHandle, BufferUsageFlags, Gpu,
    MemoryDomain,
};
use uuid::Uuid;

use crate::material_v2::*;
use crate::{ensure_vec_length, AssetMap, PipelineTarget};

use super::SamplerAllocator;

pub(crate) trait MaterialData {
    fn bind(
        &self,
        gpu: &dyn Gpu,
        material: &Material2,
        permutation: PipelineTarget,
        render_pass: &mut RenderPass2,
        asset_map: &AssetMap,
        sampler_allocator: &SamplerAllocator,
    ) -> anyhow::Result<()>;
}

#[derive(Default)]
pub(crate) struct MaterialDataManager {
    data: HashMap<Uuid, MaterialDataInfo>,
}

struct MaterialDataInfo {
    data: SparseMaterialData,
    last_update_tick: u128,
}

pub(crate) struct SparseMaterialData {
    parameter_buffer: Option<BufferHandle>,
    textures: HashMap<String, MaterialTextureLayout>,
    uniforms: HashMap<String, MaterialUniformVariableLayout>,
}

struct MaterialUniformVariableLayout {
    offset: usize,
    size: usize,
    set: u32,
    binding: u32,
}

struct MaterialTextureLayout {
    set: u32,
    binding: u32,
}

impl MaterialDataManager {
    pub(crate) fn get_data(
        &mut self,
        gpu: &dyn Gpu,
        material: &Material2,
        asset_map: &AssetMap,
    ) -> anyhow::Result<&dyn MaterialData> {
        let info: Result<&mut MaterialDataInfo, anyhow::Error> =
            match self.data.entry(material.uuid) {
                Entry::Occupied(e) => Ok(e.into_mut()),
                Entry::Vacant(slot) => {
                    let data = Self::init_material_backing_data(material, gpu, asset_map)?;
                    let info = MaterialDataInfo {
                        data,
                        last_update_tick: material.last_tick_change,
                    };
                    Ok(slot.insert(info))
                }
            };
        let info = info?;
        if info.last_update_tick != material.last_tick_change {
            Self::update_material_data(material, info, gpu, asset_map)?;
        }
        Ok(&info.data)
    }

    fn update_material_data(
        material: &Material2,
        info: &mut MaterialDataInfo,
        gpu: &dyn Gpu,
        asset_map: &AssetMap,
    ) -> anyhow::Result<()> {
        if let Some(buffer) = info.data.parameter_buffer {
            gpu.destroy_buffer(buffer);
        }

        let new_data = Self::init_material_backing_data(material, gpu, asset_map)?;

        info.data = new_data;

        let buffer = if let Some(buffer) = info.data.parameter_buffer {
            buffer
        } else {
            return Ok(());
        };

        for (name, parameter) in &material.parameters {
            if let Some(layout) = info.data.uniforms.get(name) {
                match parameter {
                    MaterialParameter::Color(col) => {
                        let bytes = bytemuck::cast_slice::<f32, u8>(col);
                        gpu.write_buffer(&buffer, layout.offset as u64, &bytes[0..layout.size])?
                    }
                    MaterialParameter::Float(val) => {
                        if layout.size != 4 {
                            anyhow::bail!("Not enough space for a float");
                        }
                        let val = [*val];
                        let bytes = bytemuck::cast_slice::<f32, u8>(&val);
                        gpu.write_buffer(&buffer, layout.offset as u64, bytes)?
                    }
                    MaterialParameter::Texture(_) => {}
                }
            }
        }

        Ok(())
    }

    fn init_material_backing_data(
        material: &Material2,
        gpu: &dyn Gpu,
        asset_map: &AssetMap,
    ) -> anyhow::Result<SparseMaterialData> {
        // 1. resolve buffer offsets
        // 2. create buffers
        // 3. write buffers at offsets
        // 4. resolve textures

        let vs_shader = asset_map.get(&material.vertex_shader);
        let fs_shader = asset_map.get(&material.fragment_shader);
        let vs_layout = gpu.get_shader_info(&vs_shader.handle)?;
        let fs_layout = gpu.get_shader_info(&fs_shader.handle)?;

        let mut uniforms = HashMap::new();
        let mut textures = HashMap::new();

        find_uniform_variables_recursive(
            "".to_owned(),
            &mut uniforms,
            &vs_layout.uniform_variables,
        );
        find_uniform_variables_recursive(
            "".to_owned(),
            &mut uniforms,
            &fs_layout.uniform_variables,
        );

        find_texture_bindings(&mut textures, &vs_layout);
        find_texture_bindings(&mut textures, &fs_layout);

        let parameter_buffer = create_parameter_buffer(material, &mut uniforms, gpu)?;

        Ok(SparseMaterialData {
            parameter_buffer,
            uniforms,
            textures,
        })
    }
}

// All parameters go into set 1 binding 0
fn create_parameter_buffer(
    material: &Material2,
    uniform_bindings: &mut HashMap<String, MaterialUniformVariableLayout>,
    gpu: &dyn Gpu,
) -> anyhow::Result<Option<BufferHandle>> {
    uniform_bindings.retain(|_k, parameter| parameter.set == 1 && parameter.binding == 1);
    if uniform_bindings.is_empty() {
        return Ok(None);
    }

    let (_, last_parameter) = uniform_bindings
        .iter()
        .max_by_key(|(_, v)| v.offset)
        .unwrap();
    let buffer_size = last_parameter.offset + last_parameter.size;
    let buffer = gpu.make_buffer(
        &BufferCreateInfo {
            label: Some(&format!("Material Parameters - {}", material.name)),
            size: buffer_size,
            usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
        },
        MemoryDomain::HostVisible,
    )?;
    Ok(Some(buffer))
}

fn find_texture_bindings(
    texture_bindings: &mut HashMap<String, MaterialTextureLayout>,
    layout: &gpu::ShaderInfo,
) {
    for (set_idx, set) in layout.descriptor_layouts.iter().enumerate() {
        for (binding_idx, binding) in set.bindings.iter().enumerate() {
            if let gpu::BindingType::CombinedImageSampler = binding.ty {
                texture_bindings.insert(
                    binding.name.clone(),
                    MaterialTextureLayout {
                        set: set_idx as u32,
                        binding: binding_idx as u32,
                    },
                );
            }
        }
    }
}

fn find_uniform_variables_recursive(
    current_scope: String,
    uniform_bindings: &mut HashMap<String, MaterialUniformVariableLayout>,
    shader_variables: &HashMap<String, UniformVariableDescription>,
) {
    for (name, var) in shader_variables {
        let current_var = current_scope.clone() + name.as_str();
        if var.inner_members.is_empty() {
            uniform_bindings.insert(
                current_var,
                MaterialUniformVariableLayout {
                    offset: var.absolute_offset as usize,
                    size: var.size as usize,
                    set: 0,
                    binding: 0,
                },
            );
        } else {
            find_uniform_variables_recursive(
                current_var + ".",
                uniform_bindings,
                &var.inner_members,
            );
        }
    }
}

impl MaterialData for SparseMaterialData {
    fn bind(
        &self,
        gpu: &dyn Gpu,
        material: &Material2,
        pipeline_target: PipelineTarget,
        render_pass: &mut RenderPass2,
        asset_map: &AssetMap,
        sampler_allocator: &SamplerAllocator,
    ) -> anyhow::Result<()> {
        let mut bindings = vec![];

        if let Some(buffer) = self.parameter_buffer {
            bindings.push(Binding2 {
                ty: gpu::DescriptorBindingType2::UniformBuffer {
                    handle: buffer,
                    offset: 0,
                    range: gpu::WHOLE_SIZE,
                },
                write: false,
            });
        }

        for (name, tex) in material.parameters.iter().filter_map(|(s, p)| {
            if let MaterialParameter::Texture(t) = p {
                Some((s, t))
            } else {
                None
            }
        }) {
            if let Some(layout) = self.textures.get(name) {
                ensure_vec_length(&mut bindings, layout.binding as usize);
                let texture = asset_map.get(tex);
                bindings[layout.binding as usize] = Binding2::image_view(
                    texture.view,
                    sampler_allocator.get(gpu, &texture.sampler_settings),
                )
            }
        }

        render_pass.bind_resources_2(MATERIAL_PARAMETER_SLOT, &bindings)?;

        render_pass.set_vertex_shader(asset_map.get(&material.vertex_shader).handle);

        if pipeline_target != PipelineTarget::DepthOnly {
            render_pass.set_fragment_shader(asset_map.get(&material.fragment_shader).handle);
        }
        Ok(())
    }
}

/*
usage
let material = load_material(...);

let material_data = get_material_data(&material);
material_data.bind(command_buffer);

*/
