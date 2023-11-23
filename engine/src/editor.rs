use crate::Plugin;

pub struct EditorPlugin {}

impl Plugin for EditorPlugin {
    fn apply(self, _app: &mut crate::BevyEcsApp) {}
}
