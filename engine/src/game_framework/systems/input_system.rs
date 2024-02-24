use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    game_framework::event_queue::EventQueue,
    input::{InputState, Key},
};

use super::System;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum InputActionState {
    #[default]
    Released,
    Pressed,
}

pub type InputBindingName = String;

#[derive(Clone, Debug)]
pub struct InputActionEvent {
    pub action_name: InputBindingName,
    pub state: InputActionState,
}

#[derive(Clone, Debug)]
pub struct InputAxisEvent {
    pub axis_name: InputBindingName,
    pub value: f32,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum InputBindingKind {
    Action,
    Axis { scale: f32 },
}

pub struct InputBinding {
    pub name: InputBindingName,
    pub kind: InputBindingKind,
}

#[derive(Default)]
pub struct InputBindings {
    key_bindings: HashMap<Key, Vec<InputBinding>>,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct InputActionDefinition {
    pub key: Key,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct InputAxisDefinition {
    pub key: Key,
    pub scale: f32,
}

#[derive(Serialize, Deserialize, Default)]
pub struct InputBindingsDefinitions {
    pub actions: HashMap<InputBindingName, Vec<InputActionDefinition>>,
    pub axes: HashMap<InputBindingName, Vec<InputAxisDefinition>>,
}

#[derive(Default)]
pub struct InputSystem {
    pub bindings: InputBindings,
}

impl InputSystem {
    pub fn new(definitions: InputBindingsDefinitions) -> Self {
        let mut bindings = InputBindings::default();
        for (name, actions) in definitions.actions {
            for action in actions {
                bindings
                    .key_bindings
                    .entry(action.key)
                    .or_default()
                    .push(InputBinding {
                        name: name.clone(),
                        kind: InputBindingKind::Action,
                    })
            }
        }

        for (name, axes) in definitions.axes {
            for axis in axes {
                bindings
                    .key_bindings
                    .entry(axis.key)
                    .or_default()
                    .push(InputBinding {
                        name: name.clone(),
                        kind: InputBindingKind::Axis { scale: axis.scale },
                    })
            }
        }

        Self { bindings }
    }
}

impl System for InputSystem {
    fn setup_resources(
        &self,
        resource_builder: &mut crate::game_framework::resources::ResourcesBuilder,
    ) {
        resource_builder.add_resource(InputState::default());
    }

    fn begin_frame(&mut self, update: super::SystemBeginFrameParams) {
        let input_state = update.resources.get::<InputState>();

        self.dispatch_events(&input_state, update.event_queue);
    }

    fn end_frame(&mut self, post_update: super::SystemEndFrameParams) {
        let mut input_state = post_update.resources.get_mut::<InputState>();
        input_state.end_frame();
    }

    fn on_os_event(&mut self, on_os_event: super::SystemOnOsEvent) {
        let mut input_state = on_os_event.resources.get_mut::<InputState>();
        input_state.update(on_os_event.event);
    }
}

impl InputActionDefinition {
    pub fn simple(key: Key) -> Self {
        Self { key }
    }
}

impl InputAxisDefinition {
    pub fn simple(key: Key) -> Self {
        Self { key, scale: 1.0 }
    }
    pub fn opposite(positive: Key, negative: Key) -> [Self; 2] {
        [
            Self {
                key: positive,
                scale: 1.0,
            },
            Self {
                key: negative,
                scale: -1.0,
            },
        ]
    }
}

impl InputBindingsDefinitions {
    pub fn define_action_bindings<S: AsRef<str>, K, I: IntoIterator<Item = K>>(
        &mut self,
        name: S,
        actions: I,
    ) -> &mut Self
    where
        K: Into<InputActionDefinition>,
    {
        self.actions
            .entry(name.as_ref().to_owned())
            .or_default()
            .extend(actions.into_iter().map(|k| k.into()));
        self
    }

    pub fn define_axis_bindings<S: AsRef<str>>(
        &mut self,
        name: S,
        axes: impl IntoIterator<Item = InputAxisDefinition>,
    ) -> &mut Self {
        self.axes
            .entry(name.as_ref().to_owned())
            .or_default()
            .extend(axes);
        self
    }
}

impl InputBindings {
    fn iter_key_to_bindings(&self) -> impl Iterator<Item = (Key, &[InputBinding])> {
        self.key_bindings.iter().map(|(k, v)| (*k, v.as_ref()))
    }
}

impl InputSystem {
    fn dispatch_events(&mut self, input_state: &InputState, event_queue: &EventQueue) {
        let mut action_events_to_dispatch = HashMap::<&InputBindingName, InputActionState>::new();
        let mut axis_events_to_dispatch = HashMap::<&InputBindingName, f32>::new();

        for (key, bindings) in self.bindings.iter_key_to_bindings() {
            for binding in bindings {
                let (can_dispatch, scale) = match binding.kind {
                    InputBindingKind::Action => (
                        input_state.is_key_just_pressed(key)
                            || input_state.is_key_just_released(key),
                        0.0,
                    ),
                    InputBindingKind::Axis { scale } => (
                        true,
                        if input_state.is_key_pressed(key) {
                            1.0 * scale
                        } else {
                            0.0
                        },
                    ),
                };
                if can_dispatch {
                    let state = if input_state.is_key_just_pressed(key) {
                        InputActionState::Pressed
                    } else {
                        InputActionState::Released
                    };
                    match binding.kind {
                        InputBindingKind::Action => {
                            action_events_to_dispatch.insert(&binding.name, state);
                        }
                        InputBindingKind::Axis { .. } => {
                            *axis_events_to_dispatch.entry(&binding.name).or_default() += scale;
                        }
                    }
                }
            }
        }

        for (action_name, state) in action_events_to_dispatch {
            event_queue.push_event(InputActionEvent {
                action_name: action_name.clone(),
                state,
            });
        }
        for (axis_name, value) in axis_events_to_dispatch {
            event_queue.push_event(InputAxisEvent {
                axis_name: axis_name.clone(),
                value,
            });
        }
    }
}

impl IntoIterator for Key {
    type Item = Key;

    type IntoIter = std::iter::Once<Key>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self)
    }
}

impl From<Key> for InputActionDefinition {
    fn from(value: Key) -> Self {
        InputActionDefinition::simple(value)
    }
}
