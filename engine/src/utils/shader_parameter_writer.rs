use glam::{Vec2, Vec3, Vec4};
use log::warn;
use mgpu::{
    Buffer, BufferDescription, BufferUsageFlags, BufferWriteParams, Device, ShaderModuleLayout,
    VariableType, VertexAttributeFormat,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum ScalarParameterType {
    Scalar(f32),
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
pub struct ScalarParameterInfo {
    pub name: String,
    pub offset: usize,
    pub ty: VertexAttributeFormat,
}

pub struct ScalarParameterWriter {
    pub parameters_infos: Vec<ScalarParameterInfo>,
    pub binary_blob: Vec<u8>,
}

impl ScalarParameterWriter {
    pub fn new(
        device: &Device,
        layouts: &[&ShaderModuleLayout],
        set: usize,
        binding: usize,
    ) -> anyhow::Result<Self> {
        let mut user_scalars = vec![];
        for layout in layouts {
            for variable in layout
                .variables
                .iter()
                .filter(|var| var.binding_index == binding && var.binding_set == set)
            {
                match variable.ty {
                    VariableType::Field { format, offset } => {
                        user_scalars.push(ScalarParameterInfo {
                            name: variable
                                .name
                                .clone()
                                .expect("A shader variable doesn't have a name"),
                            ty: format,
                            offset,
                        });
                    }
                    _ => panic!("For now only fields are supported"),
                };
            }
        }

        let size = user_scalars
            .iter()
            .max_by_key(|v| v.offset)
            .map(|v| v.offset + std::mem::size_of::<[f32; 4]>())
            .unwrap_or_default();

        Ok(Self {
            parameters_infos: user_scalars,
            binary_blob: vec![0u8; size],
        })
    }

    pub fn write(&mut self, param_name: &str, value: impl Into<ScalarParameterType>) {
        let Some(parameter) = self.parameters_infos.iter().find(|p| p.name == param_name) else {
            return;
        };

        debug_assert!(
            parameter.offset + parameter.ty.size_bytes() <= self.binary_blob.len(),
            "Parameter {} at offset {} of size {} goes out of data buffer!",
            param_name,
            parameter.offset,
            parameter.ty.size_bytes()
        );

        let value = value.into();

        if value.ty() != parameter.ty {
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
                ScalarParameterType::Vec2(val) => ptr.cast::<[f32; 2]>().write(val.to_array()),
                ScalarParameterType::Vec3(val) => ptr.cast::<[f32; 3]>().write(val.to_array()),
                ScalarParameterType::Vec4(val) => ptr.cast::<[f32; 4]>().write(val.to_array()),
            }
        }
    }

    pub fn update_buffer(&self, device: &Device, buffer: Buffer) -> anyhow::Result<()> {
        debug_assert!(buffer.size() >= self.binary_blob.len());
        device.write_buffer(
            buffer,
            &BufferWriteParams {
                data: &self.binary_blob,
                offset: 0,
                size: buffer.size(),
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
