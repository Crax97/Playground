use egui_mgpu::egui::{self, Ui};

use crate::{
    asset_map::{Asset, AssetHandle, AssetMap},
    immutable_string::ImmutableString,
};

#[derive(Default)]
pub struct AssetPicker {
    is_shown: bool,
    selected_asset: Option<ImmutableString>,
}

impl AssetPicker {
    pub fn draw<A: Asset>(
        &mut self,
        ui: &mut Ui,
        handle: &mut AssetHandle<A>,
        asset_map: &mut AssetMap,
    ) {
        let label = handle.identifier().to_string();

        ui.horizontal(|ui| {
            ui.label(&label);
            if ui.button("Select").clicked() {
                self.is_shown = true;
            }
        });

        if self.is_shown() {
            egui::Window::new("asset_picker").show(ui.ctx(), |ui| {
                ui.group(|ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for id in asset_map.iter_ids::<A>() {
                            if ui.button(id.to_string()).clicked() {
                                self.selected_asset = Some(id.clone());
                            }
                        }
                    });
                });

                ui.group(|ui| {
                    if ui
                        .add_enabled(self.selected_asset.is_some(), egui::Button::new("Confirm"))
                        .clicked()
                    {
                        let _ = std::mem::replace(
                            handle,
                            AssetHandle::new(self.selected_asset.take().unwrap()),
                        );
                        self.is_shown = false;
                    };

                    if ui.button("Cancel").clicked() {
                        self.is_shown = false;
                        self.selected_asset.take();
                    }
                });
            });
        }
    }

    pub fn is_shown(&self) -> bool {
        self.is_shown
    }
}
