//! The old graph paths must keep resolving after the move to `tinyagents-tasks`.

use crate::orchestration::{
    DetachedTaskRegistry, InMemoryTaskStore, JsonlTaskStore, OrchestrationTaskStatus,
    SteeringRegistry, TaskStore, TaskStoreRegistry, reconcile_orphaned_tasks,
};

#[test]
fn old_graph_paths_are_the_tasks_crate_types() {
    fn same<T>(_: std::marker::PhantomData<T>, _: std::marker::PhantomData<T>) {}
    same(
        std::marker::PhantomData::<crate::OrchestrationTaskStatus>,
        std::marker::PhantomData::<tinyagents_tasks::OrchestrationTaskStatus>,
    );
    same(
        std::marker::PhantomData::<OrchestrationTaskStatus>,
        std::marker::PhantomData::<tinyagents_tasks::OrchestrationTaskStatus>,
    );
    same(
        std::marker::PhantomData::<SteeringRegistry>,
        std::marker::PhantomData::<tinyagents_tasks::SteeringRegistry>,
    );
    let _ = std::marker::PhantomData::<(
        InMemoryTaskStore,
        JsonlTaskStore,
        TaskStoreRegistry<u8>,
        DetachedTaskRegistry<(), ()>,
        Box<dyn TaskStore>,
    )>;
    let _ = reconcile_orphaned_tasks;
}
