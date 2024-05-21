use glam::{Vec2, Vec3, Vec4};
use log::warn;
use mgpu::{
    Buffer, BufferWriteParams, Device, ShaderModuleLayout,
    VariableType, VertexAttributeFormat,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum ScalarParameterType {
    Scalar(f32),
    ScalarU32(u32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ScalarMaterialParameter {
    pub name: String,
    pub value: ScalarParameterType,
}

#[derive(Debug)]
pub enum ScalarParameterTy {
    Field(VertexAttributeFormat),
    Array {
        layout: VariableType,
        length: Option<usize>,
    },
}

#[derive(Debug)]
pub struct ScalarParameterInfo {
    pub name: String,
    pub offset: usize,
    pub ty: ScalarParameterTy,
}

pub struct ScalarParameterWriter {
    pub scalar_infos: Vec<ScalarParameterInfo>,
    pub binary_blob: Vec<u8>,
}

impl ScalarParameterWriter {
    pub fn new(
        _device: &Device,
        layouts: &[&ShaderModuleLayout],
        set: usize,
        binding: usize,
    ) -> anyhow::Result<Self> {
        let mut scalar_infos = vec![];
        for layout in layouts {
            for variable in layout
                .variables
                .iter()
                .filter(|var| var.binding_index == binding && var.binding_set == set)
            {
                match &variable.ty {
                    VariableType::Field { format } => {
                        scalar_infos.push(ScalarParameterInfo {
                            name: variable
                                .name
                                .clone()
                                .expect("a shader variable doesn't have a name"),
                            ty: ScalarParameterTy::Field(*format),
                            offset: variable.offset,
                        });
                    }
                    VariableType::Array {
                        members_layout,
                        length,
                    } => scalar_infos.push(ScalarParameterInfo {
                        name: variable
                            .name
                            .clone()
                            .expect("A shader variable doesnt have a name"),
                        offset: variable.offset,
                        ty: ScalarParameterTy::Array {
                            layout: *members_layout.clone(),
                            length: *length,
                        },
                    }),
                    _ => panic!("For now only fields are supported: {:#?}", variable),
                };
            }
        }

        let size = scalar_infos
            .iter()
            .max_by_key(|v| v.offset)
            .map(|v| {
                let size_elem = match &v.ty {
                    ScalarParameterTy::Field(_) => std::mem::size_of::<[f32; 4]>(),
                    ScalarParameterTy::Array { layout, length } => {
                        layout.size() * length.unwrap_or(1)
                    }
                };
                v.offset + size_elem
            })
            .unwrap_or_default();

        Ok(Self {
            scalar_infos,
            binary_blob: vec![0u8; size],
        })
    }

    pub fn write(&mut self, param_name: &str, value: impl Into<ScalarParameterType>) {
        let Some(parameter) = self.scalar_infos.iter().find(|p| p.name == param_name) else {
            return;
        };

        match &parameter.ty {
            ScalarParameterTy::Field(ty) => {
                debug_assert!(
                    parameter.offset + ty.size_bytes() <= self.binary_blob.len(),
                    "Parameter {} at offset {} of size {} goes out of data buffer!",
                    param_name,
                    parameter.offset,
                    ty.size_bytes()
                );

                let value = value.into();

                if value.ty() != *ty {
                    warn!(
                "Tried setting material parameter {} of type {:?} but got a value of type {:?}",
                parameter.name,
                parameter.ty,
                value.ty()
            );
                    return;
                }

                let ptr = unsafe { self.binary_blob.as_mut_ptr().add(parameter.offset) };

                // We checked that the parameter doesn't write outside of the binary blob +
                // that the parameter is of the same type as the one in the shader
                unsafe {
                    match value {
                        ScalarParameterType::Scalar(val) => ptr.cast::<f32>().write(val),
                        ScalarParameterType::ScalarU32(val) => ptr.cast::<u32>().write(val),
                        ScalarParameterType::Vec2(val) => {
                            ptr.cast::<[f32; 2]>().write(val.to_array())
                        }
                        ScalarParameterType::Vec3(val) => {
                            ptr.cast::<[f32; 3]>().write(val.to_array())
                        }
                        ScalarParameterType::Vec4(val) => {
                            ptr.cast::<[f32; 4]>().write(val.to_array())
                        }
                    }
                }
            }
            ScalarParameterTy::Array { .. } => {
                panic!("Cannot write an array using write()");
            }
        }
    }

    pub fn write_array(&mut self, array_name: &str, value: &[u8]) {
        let Some(parameter) = self.scalar_infos.iter().find(|p| p.name == array_name) else {
            return;
        };

        match &parameter.ty {
            ScalarParameterTy::Field(_) => panic!("Cannot write a field as an array"),
            ScalarParameterTy::Array { layout, length } => {
                let layout_size = layout.size();
                debug_assert!(
                    value.len() % layout_size == 0,
                    "Writing to array {} bytes, the layout size is {}, rem is {}",
                    value.len(),
                    layout_size,
                    value.len() % layout_size
                );
                if let Some(length) = length {
                    debug_assert!(value.len() >= layout_size * length);
                }

                let num_bytes = length.unwrap_or(value.len() / layout_size) * layout_size;
                let final_write_offset = parameter.offset + num_bytes;
                if self.binary_blob.len() < final_write_offset {
                    self.binary_blob.resize(final_write_offset, 0);
                }

                unsafe {
                    let base = self.binary_blob.as_mut_ptr().add(parameter.offset);
                    base.copy_from(value.as_ptr(), num_bytes);
                }
            }
        };
    }

    pub fn update_buffer(&self, device: &Device, buffer: Buffer) -> anyhow::Result<()> {
        debug_assert!(buffer.size() >= self.binary_blob.len());
        device.write_buffer(
            buffer,
            &BufferWriteParams {
                data: &self.binary_blob,
                offset: 0,
                size: self.binary_blob.len(),
            },
        )?;
        Ok(())
    }
}

impl ScalarParameterType {
    fn ty(&self) -> VertexAttributeFormat {
        match self {
            ScalarParameterType::Scalar(_) => VertexAttributeFormat::Float,
            ScalarParameterType::Vec2(_) => VertexAttributeFormat::Float2,
            ScalarParameterType::Vec3(_) => VertexAttributeFormat::Float3,
            ScalarParameterType::Vec4(_) => VertexAttributeFormat::Float4,
            ScalarParameterType::ScalarU32(_) => VertexAttributeFormat::Uint,
        }
    }
}

impl From<f32> for ScalarParameterType {
    fn from(value: f32) -> Self {
        ScalarParameterType::Scalar(value)
    }
}
impl From<[f32; 2]> for ScalarParameterType {
    fn from(value: [f32; 2]) -> Self {
        ScalarParameterType::Vec2(Vec2::from_array(value))
    }
}
impl From<[f32; 3]> for ScalarParameterType {
    fn from(value: [f32; 3]) -> Self {
        ScalarParameterType::Vec3(Vec3::from_array(value))
    }
}
impl From<[f32; 4]> for ScalarParameterType {
    fn from(value: [f32; 4]) -> Self {
        ScalarParameterType::Vec4(Vec4::from_array(value))
    }
}
