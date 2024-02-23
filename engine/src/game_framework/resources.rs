use std::{
    any::{Any, TypeId},
    collections::HashMap,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

pub trait Resource: 'static {}

impl<R: Send + Sync + 'static> Resource for R {}

#[derive(Clone)]
pub struct Resources {
    resources: HashMap<TypeId, Arc<RwLock<dyn AnyResource>>>,
}

#[derive(Default)]
pub struct ResourcesBuilder {
    resources: HashMap<TypeId, Arc<RwLock<dyn AnyResource>>>,
}

pub struct Ref<'a, R: Resource> {
    _ph: PhantomData<R>,
    res: RwLockReadGuard<'a, dyn AnyResource>,
}

pub struct RefMut<'a, R: Resource> {
    _ph: PhantomData<R>,
    res: RwLockWriteGuard<'a, dyn AnyResource>,
}

impl ResourcesBuilder {
    pub fn add_resource<R: Resource>(&mut self, res: R) -> &mut Self {
        self.resources
            .insert(TypeId::of::<R>(), Arc::new(RwLock::new(res)));
        self
    }

    pub fn build(self) -> Resources {
        Resources {
            resources: self.resources,
        }
    }
}

pub trait AnyResource: Any + Resource {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<R: Resource> AnyResource for R {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl<'a, R: Resource> Deref for Ref<'a, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.res.as_any().downcast_ref().unwrap()
    }
}

impl<'a, R: Resource> Deref for RefMut<'a, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.res.as_any().downcast_ref().unwrap()
    }
}

impl<'a, R: Resource> DerefMut for RefMut<'a, R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.res.as_any_mut().downcast_mut().unwrap()
    }
}

impl Resources {
    pub fn add_resource<R: Resource>(&mut self, resource: R) {
        self.resources
            .insert(TypeId::of::<R>(), Arc::new(RwLock::new(resource)));
    }

    pub fn get<R: Resource>(&self) -> Ref<'_, R> {
        self.try_get().unwrap()
    }

    pub fn try_get<R: Resource>(&self) -> Option<Ref<'_, R>> {
        self.resources.get(&TypeId::of::<R>()).map(|o| {
            let res = o.read().unwrap();
            Ref {
                _ph: PhantomData,
                res,
            }
        })
    }

    pub fn get_mut<R: Resource>(&self) -> RefMut<'_, R> {
        self.try_get_mut().unwrap()
    }

    pub fn try_get_mut<R: Resource>(&self) -> Option<RefMut<'_, R>> {
        self.resources.get(&TypeId::of::<R>()).map(|o| {
            let res = o.write().unwrap();

            RefMut {
                _ph: PhantomData,
                res,
            }
        })
    }
}
