use egui::{emath::Numeric, Rect, Response, Sense, Ui, Vec2, Widget, WidgetText};

pub trait UiExtension {
    fn edit_numbers<N: Numeric>(&mut self, numbers: &mut [N]) -> bool;
    fn edit_number<N: Numeric>(&mut self, number: &mut N) -> Response;
    fn horizontal_with_label<R, F: FnOnce(&mut Ui) -> R>(
        &mut self,
        label: impl Into<WidgetText>,
        add_contents: F,
    );
    fn fill_width(&mut self) -> Rect;
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
    fn edit_numbers<N: Numeric>(&mut self, numbers: &mut [N]) -> bool {
        let mut cum = false;
        self.horizontal(|ui| {
            for num in numbers {
                cum |= egui::DragValue::new(num).ui(ui).changed();
            }
        });
        cum
    }
    fn edit_number<N: Numeric>(&mut self, number: &mut N) -> Response {
        self.add(egui::DragValue::new(number).update_while_editing(false))
    }
    fn fill_width(&mut self) -> Rect {
        self.allocate_at_least(
            Vec2 {
                x: self.available_width(),
                y: 0.0,
            },
            Sense::focusable_noninteractive(),
        )
        .0
    }
}
