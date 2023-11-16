use imgui::{InputTextCallback, InputTextCallbackHandler, InputTextFlags, Ui};

use crate::{
    input::{key::Key, InputState},
    CvarManager,
};

pub struct ImguiConsole {
    show: bool,
    messages: Vec<String>,
    pending_input: String,
}

impl ImguiConsole {
    pub fn new() -> Self {
        Self {
            show: false,
            messages: vec![],
            pending_input: String::new(),
        }
    }
    pub fn update(&mut self, input: &InputState) {
        if input.is_key_just_pressed(Key::F8) {
            self.show = !self.show;
        }
    }
    pub fn add_message<S: AsRef<str>>(&mut self, message: S) {
        if message.as_ref().is_empty() {
            return;
        }
        log::info!("{}", message.as_ref());
        self.messages.push(message.as_ref().to_owned());
    }
    pub fn imgui_update(&mut self, ui: &Ui, cvar_manager: &mut CvarManager) {
        struct ConsoleCallback<'a>(&'a mut CvarManager);
        impl<'a> InputTextCallbackHandler for ConsoleCallback<'a> {
            fn on_completion(&mut self, mut data: imgui::TextCallbackData) {
                let content = data.str();
                for cvar in self.0.cvar_names() {
                    if cvar.contains(content) {
                        data.clear();
                        data.push_str(cvar);
                        return;
                    }
                }
            }

            fn on_history(&mut self, _: imgui::HistoryDirection, _: imgui::TextCallbackData) {}
        }
        if self.show {
            let display_size = ui.io().display_size;
            let console_size = [display_size[0], display_size[1] / 3.0];
            let window = ui
                .window("Console")
                .position_pivot([0.0, 0.0])
                .position([0.0, 0.0], imgui::Condition::Always)
                .size(console_size, imgui::Condition::Always)
                .resizable(false)
                .collapsible(false)
                .title_bar(false)
                .begin()
                .unwrap();
            ui.set_window_font_scale(0.8);
            let output = ui
                .child_window("Output")
                .size([0.0, -ui.text_line_height() * 2.0])
                .begin()
                .unwrap();
            for message in &self.messages {
                ui.text(message);
            }
            ui.set_scroll_here_y();
            output.end();

            let w = ui.push_item_width(console_size[0]);
            ui.set_keyboard_focus_here();
            if ui
                .input_text("ConsoleInput", &mut self.pending_input)
                .flags(InputTextFlags::ENTER_RETURNS_TRUE | InputTextFlags::CALLBACK_RESIZE)
                .callback(
                    InputTextCallback::COMPLETION | InputTextCallback::HISTORY,
                    ConsoleCallback(cvar_manager),
                )
                .build()
            {
                let input = std::mem::take(&mut self.pending_input);
                self.add_message(input.clone());
                self.handle_cvar_command(&input, cvar_manager);
            }
            w.end();
            window.end();
        }
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
