use egui::{emath::Numeric, Rect, Response, Sense, Ui, Vec2, Widget, WidgetText};

pub trait UiExtension {
    fn slider<N: Numeric>(&mut self, label: &str, min: N, max: N, current: &mut N);
    fn edit_numbers<N: Numeric>(&mut self, numbers: &mut [N]) -> bool;
    fn edit_number<N: Numeric>(&mut self, number: &mut N) -> Response;
    fn horizontal_with_label<R, F: FnOnce(&mut Ui) -> R>(
        &mut self,
        label: impl Into<WidgetText>,
        add_contents: F,
    ) -> R;
    fn fill_width(&mut self) -> Rect;
    fn color_edit3(&mut self, label: &str, values: &mut [f32; 3]);
    fn color_edit4(&mut self, label: &str, values: &mut [f32; 4]);
    fn input_float(&mut self, label: &str, value: &mut f32) -> bool;
    fn input_floats(&mut self, label: &str, values: &mut [f32]) -> bool;
}

impl UiExtension for Ui {
    fn slider<N: Numeric>(&mut self, label: &str, min: N, max: N, current: &mut N) {
        self.horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(current).clamp_range(min..=max));
        });
    }
    fn horizontal_with_label<R, F: FnOnce(&mut Ui) -> R>(
        &mut self,
        label: impl Into<WidgetText>,
        add_contents: F,
    ) -> R {
        self.horizontal(|ui| {
            ui.label(label);
            add_contents(ui)
        })
        .inner
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

    fn color_edit3(&mut self, label: &str, values: &mut [f32; 3]) {
        self.horizontal(|ui| {
            ui.label(label);
            ui.color_edit_button_rgb(values);
        });
    }

    fn color_edit4(&mut self, label: &str, values: &mut [f32; 4]) {
        self.horizontal(|ui| {
            ui.label(label);
            ui.color_edit_button_rgba_unmultiplied(values);
        });
    }

    fn input_float(&mut self, label: &str, value: &mut f32) -> bool {
        self.horizontal_with_label(label, |ui| ui.edit_number(value))
            .changed()
    }

    fn input_floats(&mut self, label: &str, values: &mut [f32]) -> bool {
        self.horizontal_with_label(label, |ui| ui.edit_numbers(values))
    }
}
