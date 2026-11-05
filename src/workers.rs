//! Persistent worker threads that run borrowed jobs.
//!
//! The search pool used to create a thread per helper at every `go` and let it die at
//! the end: thread creation and NUMA binding, both of which can fail, were redone on
//! every move, and a refused creation aborted the engine in the middle of a game. The
//! workers here are created once, when the number of threads is set, bound once, and
//! handed a job for each search, table build or clear.
//!
//! [`Workers::scope`] works like [`std::thread::scope`]: jobs may borrow from the
//! caller, because the scope does not return before every job it started has finished.

use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;
type Payload = Box<dyn Any + Send + 'static>;

/// What a worker is doing, as its owner and the worker itself see it.
enum Slot {
    /// Waiting for a job.
    Idle,
    /// Handed a job it has not picked up yet.
    Pending(Job),
    /// Running its job.
    Running,
    /// Done; carries the panic of the job, if it panicked.
    Finished(Option<Payload>),
    /// Told to end its thread.
    Exit,
}

struct Channel {
    slot: Mutex<Slot>,
    changed: Condvar,
}

impl Channel {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        // Nothing panics with the lock held: a poisoned lock still holds a valid slot.
        self.slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, Slot>) -> MutexGuard<'a, Slot> {
        self.changed.wait(guard).unwrap_or_else(|e| e.into_inner())
    }

    /// Hand over a job. The worker must be idle.
    fn give(&self, job: Job) {
        let mut slot = self.lock();
        debug_assert!(matches!(*slot, Slot::Idle), "a worker given two jobs at once");
        *slot = Slot::Pending(job);
        drop(slot);
        self.changed.notify_all();
    }

    /// Wait for the job handed over to finish, leave the worker idle, and return the
    /// panic of the job if it panicked.
    fn join(&self) -> Option<Payload> {
        let mut slot = self.lock();
        loop {
            match std::mem::replace(&mut *slot, Slot::Idle) {
                Slot::Finished(panic) => return panic,
                other => {
                    debug_assert!(!matches!(other, Slot::Idle | Slot::Exit), "joined without a job");
                    *slot = other;
                    slot = self.wait(slot);
                }
            }
        }
    }
}

fn worker_loop(channel: Arc<Channel>) {
    loop {
        let job = {
            let mut slot = channel.lock();
            loop {
                match std::mem::replace(&mut *slot, Slot::Running) {
                    Slot::Pending(job) => break job,
                    Slot::Exit => return,
                    other => {
                        *slot = other;
                        slot = channel.wait(slot);
                    }
                }
            }
        };
        let panic = catch_unwind(AssertUnwindSafe(job)).err();
        *channel.lock() = Slot::Finished(panic);
        channel.changed.notify_all();
    }
}

struct Worker {
    channel: Arc<Channel>,
    handle: Option<JoinHandle<()>>,
}

/// A fixed set of persistent threads, addressed by index.
pub struct Workers {
    workers: Vec<Worker>,
}

#[cfg(test)]
thread_local! {
    /// Lets a test make the `n`-th creation of a worker thread fail, as a system short
    /// of threads or memory would. Per thread, so that tests running side by side do
    /// not see each other's.
    pub(crate) static FAIL_SPAWN_AT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

impl Workers {
    /// No thread at all: every job runs on the caller.
    pub const fn none() -> Self {
        Workers {
            workers: Vec::new(),
        }
    }

    /// Start up to `count` threads with `stack` bytes of stack each, named
    /// `{name}-{i}`. Stops at the first creation the system refuses: [`Workers::len`]
    /// says how many there are, and the caller works with those.
    pub fn spawn(count: usize, stack: usize, name: &str) -> Self {
        let mut workers = Vec::with_capacity(count);
        if cfg!(target_arch = "wasm32") {
            return Workers { workers };
        }
        for i in 0..count {
            #[cfg(test)]
            if i == FAIL_SPAWN_AT.get() {
                break;
            }
            let channel = Arc::new(Channel {
                slot: Mutex::new(Slot::Idle),
                changed: Condvar::new(),
            });
            let for_thread = Arc::clone(&channel);
            match std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .stack_size(stack)
                .spawn(move || worker_loop(for_thread))
            {
                Ok(handle) => workers.push(Worker {
                    channel,
                    handle: Some(handle),
                }),
                Err(_) => break,
            }
        }
        Workers { workers }
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Run jobs that borrow from the caller on the workers, and wait for all of them.
    ///
    /// A job that panics has its panic resumed here, once every job has finished. The
    /// scope waits for its jobs even when `f` itself panics, which is what makes the
    /// borrows sound.
    pub fn scope<'env, F, T>(&self, f: F) -> T
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
    {
        let scope = Scope {
            workers: self,
            started: RefCell::new(Vec::new()),
            _scope: PhantomData,
            _env: PhantomData,
        };
        // Caught rather than left to unwind: the jobs must be waited for first.
        let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));
        let job_panic = scope.join_all();
        match result {
            Err(panic) => resume_unwind(panic),
            Ok(_) if job_panic.is_some() => resume_unwind(job_panic.unwrap()),
            Ok(value) => value,
        }
    }

    /// Start a job that borrows nothing on worker `i`, to be waited for with
    /// [`Workers::join`]. For work that outlives any one scope of its owner.
    pub fn launch(&self, i: usize, job: impl FnOnce() + Send + 'static) {
        self.workers[i].channel.give(Box::new(job));
    }

    /// Wait for the job started by [`Workers::launch`] on worker `i`.
    pub fn join(&self, i: usize) {
        if let Some(panic) = self.workers[i].channel.join() {
            resume_unwind(panic);
        }
    }

    /// The OS thread of worker `i`, for tests that check a thread is reused.
    #[cfg(test)]
    pub(crate) fn thread_id(&self, i: usize) -> std::thread::ThreadId {
        self.workers[i].handle.as_ref().expect("a running worker").thread().id()
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        for worker in &self.workers {
            let mut slot = worker.channel.lock();
            debug_assert!(matches!(*slot, Slot::Idle), "workers dropped with a job running");
            *slot = Slot::Exit;
            drop(slot);
            worker.channel.changed.notify_all();
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The jobs of one [`Workers::scope`]: each worker runs at most one at a time.
pub struct Scope<'scope, 'env: 'scope> {
    workers: &'scope Workers,
    started: RefCell<Vec<usize>>,
    _scope: PhantomData<&'scope mut &'scope ()>,
    _env: PhantomData<&'env mut &'env ()>,
}

impl<'scope, 'env> Scope<'scope, 'env> {
    /// Run `job` on worker `i`. It may borrow anything that outlives the scope.
    pub fn run<F>(&self, i: usize, job: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        debug_assert!(i < self.workers.len(), "worker {i} of {}", self.workers.len());
        debug_assert!(!self.started.borrow().contains(&i), "worker {i} given two jobs in one scope");
        let job: Box<dyn FnOnce() + Send + 'scope> = Box::new(job);
        // SAFETY: `Workers::scope` joins every job it started before it returns, even
        // when its closure panics, so nothing the job borrows can end while it runs:
        // the lifetime is only erased to cross the channel.
        let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Job>(job) };
        self.started.borrow_mut().push(i);
        self.workers.workers[i].channel.give(job);
    }

    /// Wait for every job started; the first panic among them, if any.
    fn join_all(&self) -> Option<Payload> {
        let started = std::mem::take(&mut *self.started.borrow_mut());
        let mut first = None;
        for i in started {
            if let Some(panic) = self.workers.workers[i].channel.join() {
                first.get_or_insert(panic);
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn jobs_borrow_from_the_caller_and_are_all_done_on_return() {
        let workers = Workers::spawn(4, 1 << 20, "test");
        assert_eq!(workers.len(), 4);
        let mut cells = vec![0u64; 4];
        let shared = AtomicUsize::new(0);
        workers.scope(|s| {
            for (i, cell) in cells.iter_mut().enumerate() {
                let shared = &shared;
                s.run(i, move || {
                    *cell = (i as u64 + 1) * 10;
                    shared.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(cells, [10, 20, 30, 40]);
        assert_eq!(shared.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn the_same_threads_run_every_scope() {
        let workers = Workers::spawn(3, 1 << 20, "test");
        let mut seen = vec![Vec::new(); 3];
        for _ in 0..5 {
            let mut ids = vec![None; 3];
            workers.scope(|s| {
                for (i, id) in ids.iter_mut().enumerate() {
                    s.run(i, move || *id = Some(std::thread::current().id()));
                }
            });
            for (i, id) in ids.into_iter().enumerate() {
                seen[i].push(id.unwrap());
            }
        }
        for (i, ids) in seen.iter().enumerate() {
            assert!(ids.iter().all(|id| *id == workers.thread_id(i)), "worker {i} changed thread");
        }
    }

    #[test]
    fn a_panicking_job_is_reported_after_the_others_finish() {
        let workers = Workers::spawn(2, 1 << 20, "test");
        let done = AtomicUsize::new(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            workers.scope(|s| {
                s.run(0, || panic!("job failed"));
                s.run(1, || {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    done.fetch_add(1, Ordering::Relaxed);
                });
            })
        }));
        assert!(result.is_err());
        assert_eq!(done.load(Ordering::Relaxed), 1, "the scope returned before its jobs");
        // The workers are still usable.
        let mut x = 0;
        workers.scope(|s| s.run(0, || x = 7));
        assert_eq!(x, 7);
    }

    #[test]
    fn a_refused_creation_leaves_the_workers_already_made() {
        FAIL_SPAWN_AT.set(2);
        let workers = Workers::spawn(5, 1 << 20, "test");
        FAIL_SPAWN_AT.set(usize::MAX);
        assert_eq!(workers.len(), 2);
    }

    #[test]
    fn a_launched_job_runs_until_joined() {
        let workers = Workers::spawn(1, 1 << 20, "test");
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let ran = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&ran);
        workers.launch(0, move || {
            let _ = rx.recv();
            r.fetch_add(1, Ordering::Relaxed);
        });
        drop(tx);
        workers.join(0);
        assert_eq!(ran.load(Ordering::Relaxed), 1);
    }
}
