use egui::{emath::Numeric, Ui, Widget, WidgetText};

pub trait UiExtension {
    fn edit_numbers<N: Numeric>(&mut self, numbers: &mut [N]);
    fn edit_number<N: Numeric>(&mut self, number: &mut N);
    fn horizontal_with_label<R, F: FnOnce(&mut Ui) -> R>(
        &mut self,
        label: impl Into<WidgetText>,
        add_contents: F,
    );
}

impl UiExtension for Ui {
    fn horizontal_with_label<R, F: FnOnce(&mut Ui) -> R>(
        &mut self,
        label: impl Into<WidgetText>,
        add_contents: F,
    ) {
        self.horizontal(|ui| {
            ui.label(label);
            add_contents(ui)
        });
    }
    fn edit_numbers<N: Numeric>(&mut self, numbers: &mut [N]) {
        self.horizontal(|ui| {
            for num in numbers {
                egui::DragValue::new(num).ui(ui);
            }
        });
    }
    fn edit_number<N: Numeric>(&mut self, number: &mut N) {
        egui::DragValue::new(number).ui(self);
    }
}
