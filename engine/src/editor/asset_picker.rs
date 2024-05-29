use egui_mgpu::egui::{self, Id, Ui};

use crate::{
    asset_map::{Asset, AssetHandle, AssetMap},
    immutable_string::ImmutableString,
};

#[derive(Default)]
pub struct AssetPicker {
    is_shown: bool,
    selected_asset: Option<ImmutableString>,
    picking_for: Option<Id>,
}

impl AssetPicker {
    // returns true if the handle was changed in this frame
    pub fn modify<A: Asset>(
        &mut self,
        ui: &mut Ui,
        handle: &mut AssetHandle<A>,
        asset_map: &mut AssetMap,
    ) -> bool {
        let unique_id = ui.next_auto_id();
        let mut picked = false;
        let label: String = handle
            .identifier()
            .map(|s| s.to_string())
            .unwrap_or("(None)".into());
        ui.horizontal(|ui| {
            ui.label(&label);

            if ui
                .add_enabled(self.picking_for.is_none(), egui::Button::new("Select"))
                .clicked()
            {
                self.is_shown = true;
                self.picking_for = Some(unique_id);
            }
        });

        if self.picking_for.as_ref().is_some_and(|id| id != &unique_id) {
            return false;
        }

        if self.is_shown() {
            egui::Window::new("Asset Picker").show(ui.ctx(), |ui| {
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
                        let new_handle = AssetHandle::new(self.selected_asset.take().unwrap());
                        let _old = std::mem::replace(handle, new_handle);
                        // dec ref
                        self.is_shown = false;
                        self.picking_for = None;
                        picked = true;
                    };

                    if ui.button("Clear").clicked() {
                        let _old = std::mem::replace(handle, AssetHandle::null());
                        // dec ref
                        self.is_shown = false;
                        self.selected_asset.take();
                        self.picking_for = None;
                        picked = true;
                    }

                    if ui.button("Cancel").clicked() {
                        self.is_shown = false;
                        self.selected_asset.take();
                        self.picking_for = None;
                    }
                });
            });
        };
        picked
    }

    pub fn is_shown(&self) -> bool {
        self.is_shown
    }
}
