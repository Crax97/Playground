use bevy_ecs::system::Resource;
use crossbeam::channel::{Receiver, Sender};

use crate::{
    input::{key::Key, InputState},
    CvarManager,
};

#[derive(Clone, Debug)]
pub struct Message {
    content: String,
}

pub struct Console {
    pub(crate) show: bool,
    pub(crate) messages: Vec<String>,
    pub(crate) max_messages: usize,
    pub(crate) pending_input: String,
    pub(crate) message_receiver: Receiver<Message>,
}

#[derive(Resource)]
pub struct ConsoleWriter {
    message_sender: Sender<Message>,
}

impl Console {
    pub fn new() -> Self {
        Self::new_with_writer().0
    }

    pub fn new_with_writer() -> (Self, ConsoleWriter) {
        let (message_sender, message_receiver) = crossbeam::channel::unbounded();

        let writer = ConsoleWriter { message_sender };
        let console = Self {
            show: false,
            messages: vec![],
            pending_input: String::new(),
            message_receiver,
            max_messages: 15,
        };

        (console, writer)
    }

    pub fn update(&mut self, input: &InputState) {
        if input.is_key_just_pressed(Key::F8) {
            self.show = !self.show;
        }

        let messages = self.message_receiver.try_iter().collect::<Vec<_>>();
        for message in messages {
            self.add_message(message.content)
        }

        while self.messages.len() > self.max_messages {
            self.messages.remove(0);
        }
    }
    pub fn add_message<S: AsRef<str>>(&mut self, message: S) {
        if message.as_ref().is_empty() {
            return;
        }
        log::info!("{}", message.as_ref());
        self.messages.push(message.as_ref().to_owned());
    }

    fn handle_cvar_command(&mut self, command: &str, cvar_manager: &mut CvarManager) {
        let commands = command.split(" ").collect::<Vec<_>>();
        if commands.len() > 1 {
            match cvar_manager.set_named(commands[0], commands[1]) {
                Ok(()) => {}
                Err(e) => self.add_message(e.to_string()),
            }
        } else {
            let cvar = cvar_manager.get_named::<String>(command);
            match cvar {
                Ok(value) => self.add_message(format!("{}", value)),
                Err(e) => {
                    self.add_message(format!("Error with previous command: {}", e));
                }
            }
        }
    }
}

impl ConsoleWriter {
    pub fn write_message<S: AsRef<str>>(&mut self, message: S) -> anyhow::Result<()> {
        self.message_sender.send(Message {
            content: message.as_ref().to_owned(),
        })?;
        Ok(())
    }
}
