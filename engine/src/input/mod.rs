#![allow(dead_code)]
pub mod key;

use glam::{vec2, UVec2, Vec2};

use std::collections::HashMap;

use strum::EnumCount;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta},
    keyboard::ModifiersState,
};

pub use self::key::*;

#[derive(Debug)]
pub struct InputState {
    current_cursor_position: PhysicalPosition<u32>,
    last_update_cursor_position: PhysicalPosition<u32>,
    current_pointer_pressure: f32,
    window_size: PhysicalSize<u32>,
    current_wheel_delta: f32,
    current_delta_mouse: Vec2,

    pointer_button_state: HashMap<MouseButton, ElementState>,
    last_button_state: HashMap<MouseButton, ElementState>,
    key_states: [bool; Key::COUNT],
    last_key_states: [bool; Key::COUNT],

    current_modifiers: ModifierSet,
    last_modifiers: ModifierSet,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            current_cursor_position: Default::default(),
            last_update_cursor_position: Default::default(),
            current_pointer_pressure: Default::default(),
            window_size: Default::default(),
            current_wheel_delta: 0.0,
            pointer_button_state: Default::default(),
            last_button_state: Default::default(),
            key_states: [false; Key::COUNT],
            last_key_states: [false; Key::COUNT],
            current_modifiers: ModifierSet::default(),
            last_modifiers: ModifierSet::default(),
            current_delta_mouse: Vec2::default(),
        }
    }

    pub fn update(&mut self, event: &winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::Resized(new_size) => self.window_size = *new_size,
            winit::event::WindowEvent::Moved(_) => {}
            winit::event::WindowEvent::CloseRequested => {}
            winit::event::WindowEvent::DroppedFile(_) => {}
            winit::event::WindowEvent::HoveredFile(_) => {}
            winit::event::WindowEvent::HoveredFileCancelled => {}
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.update_keyboard_state(event);
            }
            winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                self.update_modifiers_state(&modifiers.state());
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let mouse_position = position.cast::<u32>();
                self.current_cursor_position = PhysicalPosition {
                    x: mouse_position.x,
                    y: self.window_size.height.saturating_sub(mouse_position.y),
                };
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                self.current_wheel_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                } / self.window_size.height as f32
                    * 0.5;
            }
            winit::event::WindowEvent::MouseInput { state, button, .. } => {
                self.set_cursor_button_state(*button, *state);
            }
            winit::event::WindowEvent::TouchpadPressure { pressure, .. } => {
                self.current_pointer_pressure = *pressure
            }
            winit::event::WindowEvent::Touch(touch) => {
                let winit::event::Touch {
                    phase,
                    location,
                    force,
                    ..
                } = touch;

                if let Some(force) = force {
                    self.current_pointer_pressure = match force {
                        winit::event::Force::Calibrated {
                            force,
                            max_possible_force,
                            ..
                        } => force / max_possible_force,
                        winit::event::Force::Normalized(force) => *force,
                    } as f32;
                }
                match phase {
                    winit::event::TouchPhase::Started => {
                        self.set_cursor_button_state(
                            winit::event::MouseButton::Left,
                            ElementState::Pressed,
                        );
                    }
                    winit::event::TouchPhase::Moved => {
                        let mouse_position = location.cast::<u32>();
                        self.current_cursor_position = PhysicalPosition {
                            x: mouse_position.x,
                            y: self.window_size.height.saturating_sub(mouse_position.y),
                        };
                    }
                    winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled => {
                        self.set_cursor_button_state(
                            winit::event::MouseButton::Left,
                            ElementState::Released,
                        );
                    }
                }
            }
            _ => {}
        };
    }

    pub fn device_event(&mut self, event: winit::event::DeviceEvent) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.current_delta_mouse = vec2(
                    delta.0 as f32 / (self.window_size.width as f32 * 0.5),
                    -1.0 * delta.1 as f32 / (self.window_size.height as f32 * 0.5),
                )
            }
            winit::event::DeviceEvent::MouseWheel {
                delta: winit::event::MouseScrollDelta::PixelDelta(px),
            } => {
                self.current_wheel_delta = px.y as f32;
            }
            _ => {}
        }
    }

    pub fn iter_all_just_pressed_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.key_states
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| {
                if b && !self.last_key_states[i] {
                    Some(i)
                } else {
                    None
                }
            })
            .map(|i| Key::try_from(i as u32).unwrap())
    }
    pub fn iter_all_just_released_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.key_states
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| {
                if !b && self.last_key_states[i] {
                    Some(i)
                } else {
                    None
                }
            })
            .map(|i| Key::try_from(i as u32).unwrap())
    }
    pub fn iter_all_pressed_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.key_states
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .map(|i| Key::try_from(i as u32).unwrap())
    }
    pub fn end_frame(&mut self) {
        self.last_button_state = self.pointer_button_state.clone();
        self.last_key_states = self.key_states;
        self.last_modifiers = self.current_modifiers;
        self.current_wheel_delta = 0.0;
        self.current_delta_mouse = vec2(0.0, 0.0);
        self.last_update_cursor_position = self.current_cursor_position;
    }

    fn set_cursor_button_state(&mut self, button: MouseButton, state: ElementState) {
        self.pointer_button_state
            .entry(button)
            .and_modify(|s| *s = state)
            .or_insert(state);
    }

    pub fn mouse_position(&self) -> Vec2 {
        Vec2::new(
            self.current_cursor_position.x as f32,
            self.current_cursor_position.y as f32,
        )
    }
    #[allow(dead_code)]
    pub fn last_position(&self) -> PhysicalPosition<u32> {
        self.last_update_cursor_position
    }
    pub fn normalized_mouse_position(&self) -> Vec2 {
        Vec2::new(
            (self.current_cursor_position.x as f32 / self.window_size.width as f32) * 2.0 - 1.0,
            -(self.current_cursor_position.y as f32 / self.window_size.height as f32) * 2.0 + 1.0,
        )
    }
    #[allow(dead_code)]
    pub fn normalized_last_mouse_position(&self) -> Vec2 {
        Vec2::new(
            (self.last_update_cursor_position.x as f32 / self.window_size.width as f32) * 2.0 - 1.0,
            (self.last_update_cursor_position.y as f32 / self.window_size.height as f32) * 2.0
                - 1.0,
        )
    }
    #[allow(dead_code)]
    pub fn mouse_delta(&self) -> Vec2 {
        self.current_delta_mouse
    }
    #[allow(dead_code)]
    pub fn normalized_mouse_delta(&self) -> Vec2 {
        self.current_delta_mouse.normalize()
    }

    pub fn is_mouse_button_just_pressed(&self, button: MouseButton) -> bool {
        match (
            self.pointer_button_state.get(&button),
            self.last_button_state.get(&button),
        ) {
            (Some(now), Some(before)) => {
                now == &ElementState::Pressed && before == &ElementState::Released
            }
            (Some(now), None) => now == &ElementState::Pressed,
            _ => false,
        }
    }
    pub fn is_mouse_button_just_released(&self, button: MouseButton) -> bool {
        match (
            self.pointer_button_state.get(&button),
            self.last_button_state.get(&button),
        ) {
            (Some(now), Some(before)) => {
                now == &ElementState::Released && before == &ElementState::Pressed
            }
            (Some(now), None) => now == &ElementState::Released,
            _ => false,
        }
    }
    #[allow(dead_code)]
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.pointer_button_state
            .get(&button)
            .map_or(false, |btn| btn == &ElementState::Pressed)
    }
    #[allow(dead_code)]
    pub fn is_mouse_button_released(&self, button: MouseButton) -> bool {
        self.pointer_button_state
            .get(&button)
            .map_or(false, |btn| btn == &ElementState::Released)
    }

    pub fn mouse_wheel_delta(&self) -> f32 {
        self.current_wheel_delta
    }

    pub fn current_pointer_pressure(&self) -> f32 {
        self.current_pointer_pressure
    }

    pub fn window_size(&self) -> UVec2 {
        UVec2::new(self.window_size.width, self.window_size.height)
    }

    pub fn is_key_just_pressed(&self, key: Key) -> bool {
        self.key_states[key as usize] && !self.last_key_states[key as usize]
    }

    pub fn is_key_just_released(&self, key: Key) -> bool {
        !self.key_states[key as usize] && self.last_key_states[key as usize]
    }

    #[allow(dead_code)]
    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.key_states[key as usize]
    }

    #[allow(dead_code)]
    pub fn is_key_released(&self, key: Key) -> bool {
        !self.is_key_pressed(key)
    }

    pub fn current_modifiers(&self) -> &ModifierSet {
        &self.current_modifiers
    }

    fn update_keyboard_state(&mut self, event: &KeyEvent) {
        match event.physical_key {
            winit::keyboard::PhysicalKey::Code(code) => {
                let key: Key = code.into();
                self.key_states[key as usize] = match event.state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                }
            }
            winit::keyboard::PhysicalKey::Unidentified(_) => todo!(),
        }
    }

    fn update_modifiers_state(&mut self, modifiers: &ModifiersState) {
        self.current_modifiers = modifiers.bits().into();
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(deprecated)] // We need ModifiersState::empty() to construct a KeyboardInput
#[cfg(test)]
mod tests {
    // use super::*;
    // use winit::event::*;

    // #[test]
    // pub fn test_key_events() {
    //     let mut input_state = InputState::new();

    //     input_state.update::<()>(&Event::WindowEvent {
    //         window_id: unsafe { winit::window::WindowId::dummy() },
    //         event: WindowEvent::KeyboardInput {
    //             device_id: unsafe { DeviceId::dummy() },
    //             input: KeyboardInput {
    //                 scancode: 0,
    //                 state: ElementState::Pressed,
    //                 virtual_keycode: Some(VirtualKeyCode::W),

    //                 modifiers: ModifiersState::empty(),
    //             },
    //             is_synthetic: false,
    //         },
    //     });

    //     assert!(input_state.is_key_just_pressed(Key::W));
    //     assert!(input_state.is_key_pressed(Key::W));
    //     assert!(input_state.is_key_released(Key::A));

    //     input_state.end_frame();
    //     input_state.update::<()>(&Event::WindowEvent {
    //         window_id: unsafe { winit::window::WindowId::dummy() },
    //         event: WindowEvent::KeyboardInput {
    //             device_id: unsafe { DeviceId::dummy() },
    //             input: KeyboardInput {
    //                 scancode: 0,
    //                 state: ElementState::Released,
    //                 virtual_keycode: Some(VirtualKeyCode::W),
    //                 modifiers: ModifiersState::empty(),
    //             },
    //             is_synthetic: false,

    //     });
    //     assert!(input_state.is_key_just_released(Key::W));
    //     assert!(input_state.is_key_released(Key::W));
    //     assert!(input_state.is_key_released(Key::A));
    //     input_state.end_frame();
    // }

    // #[test]
    // pub fn test_modifiers() {
    //     let mut input_state = InputState::new();

    //     input_state.update::<()>(&Event::WindowEvent {
    //         window_id: unsafe { winit::window::WindowId::dummy() },
    //         event: WindowEvent::ModifiersChanged(ModifiersState::from_bits_truncate(
    //             ModifiersState::SHIFT.bits() | ModifiersState::CTRL.bits(),
    //         )),
    //     });

    //     assert_eq!(
    //         input_state.current_modifiers(),
    //         &ModifierSet::new(true, false, true, false)
    //     );
    //     assert_ne!(
    //         input_state.current_modifiers(),
    //         &ModifierSet::new(true, true, true, false)
    //     );

    //     input_state.update::<()>(&Event::WindowEvent {
    //         window_id: unsafe { winit::window::WindowId::dummy() },
    //         event: WindowEvent::ModifiersChanged(ModifiersState::empty()),
    //     });

    //     assert_eq!(
    //         input_state.current_modifiers(),
    //         &ModifierSet::new(false, false, false, false)
    //     );
    //     assert_ne!(
    //         input_state.current_modifiers(),
    //         &ModifierSet::new(true, false, false, false)
    //     );
    // }
}
