mod scene;

pub use scene::*;

#[cfg(test)]
mod tests {
    use approx::*;
    use nalgebra::{point, vector, UnitQuaternion, UnitVector3};

    use crate::components::Transform;

    use super::Scene;

    type VoidScene = Scene<()>;

    #[test]
    fn test_translation_parent() {
        let mut graph = VoidScene::new();

        let child_node = graph.add_node_defaulted().build();
        graph.set_transform(
            child_node,
            Transform {
                position: point![10.0, 0.0, 0.0],
                ..Default::default()
            },
            super::TransformSpace::World,
        );

        let parent_node = graph.add_node_defaulted().build();
        graph.set_parent(child_node, Some(parent_node));

        assert!(graph
            .get_parent(child_node)
            .is_some_and(|p| p == parent_node));
        assert!(graph
            .get_children(parent_node)
            .is_some_and(|mut c| c.any(|c| *c == child_node)));

        graph.set_transform(
            parent_node,
            Transform {
                position: point![10.0, 0.0, 0.0],
                ..Default::default()
            },
            crate::game_scene::TransformSpace::World,
        );
        assert_eq!(
            graph
                .get_position(child_node, crate::game_scene::TransformSpace::World)
                .unwrap(),
            point![20.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_rotation_parent_child() {
        let mut graph = VoidScene::new();

        let child_node = graph.add_node_defaulted().build();
        graph.set_transform(
            child_node,
            Transform {
                position: point![10.0, 0.0, 0.0],
                ..Default::default()
            },
            super::TransformSpace::World,
        );

        let parent_node = graph.add_node_defaulted().build();
        graph.set_parent(child_node, Some(parent_node));

        assert!(graph
            .get_parent(child_node)
            .is_some_and(|p| p == parent_node));
        assert!(graph
            .get_children(parent_node)
            .is_some_and(|mut c| c.any(|c| *c == child_node)));

        graph.set_transform(
            parent_node,
            Transform {
                position: point![0.0, 0.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(
                    &UnitVector3::new_normalize(vector![0.0, 0.0, 1.0]),
                    90.0f32.to_radians(),
                ),
                ..Default::default()
            },
            crate::game_scene::TransformSpace::World,
        );
        assert_relative_eq!(
            graph
                .get_position(child_node, crate::game_scene::TransformSpace::World)
                .unwrap(),
            point![0.0, 10.0, 0.0]
        );
    }

    #[test]
    fn test_translation_local() {
        let mut graph = VoidScene::new();

        let child_node = graph.add_node_defaulted().build();
        graph.set_transform(
            child_node,
            Transform {
                position: point![10.0, 0.0, 0.0],
                ..Default::default()
            },
            super::TransformSpace::World,
        );

        let parent_node = graph.add_node_defaulted().build();
        graph.set_parent(child_node, Some(parent_node));

        assert!(graph
            .get_parent(child_node)
            .is_some_and(|p| p == parent_node));
        assert!(graph
            .get_children(parent_node)
            .is_some_and(|mut c| c.any(|c| *c == child_node)));

        graph.set_transform(
            parent_node,
            Transform {
                position: point![10.0, 0.0, 0.0],
                ..Default::default()
            },
            crate::game_scene::TransformSpace::World,
        );

        graph.set_transform(
            child_node,
            Transform {
                position: point![-10.0, 0.0, 0.0],
                ..Default::default()
            },
            crate::game_scene::TransformSpace::Local,
        );
        assert_eq!(
            graph
                .get_position(child_node, crate::game_scene::TransformSpace::World)
                .unwrap(),
            point![0.0, 0.0, 0.0]
        );
    }
}
