// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime services
//!
//! The `rt` module provides a narrow set of runtime services,
//! including the global heap (exported in `heap`) and unwinding and
//! backtrace support. The APIs in this module are highly unstable,
//! and should be considered as private implementation details for the
//! time being.

#![unstable(feature = "std_misc")]

// FIXME: this should not be here.
#![allow(missing_docs)]

use env;
use mem;
use prelude::v1::*;
use rt;
use sys;
use thunk::Thunk;
use sys_common::thread_info;

// Reexport some of our utilities which are expected by other crates.
pub use self::util::{default_sched_threads, min_stack, running_on_valgrind};
pub use self::unwind::{begin_unwind, begin_unwind_fmt};

// Reexport some functionality from liballoc.
pub use alloc::heap;

// Simple backtrace functionality (to print on panic)
pub mod backtrace;

// Internals
#[macro_use]
mod macros;

// These should be refactored/moved/made private over time
pub mod util;
pub mod unwind;
pub mod args;

mod at_exit_imp;
mod libunwind;

/// The default error code of the rust runtime if the main thread panics instead
/// of exiting cleanly.
pub const DEFAULT_ERROR_CODE: int = 101;

/// Ignore SIGPIPE signals.
#[cfg(unix)]
unsafe fn ignore_sigpipe() {
    use libc;
    use libc::funcs::posix01::signal::signal;
    assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != -1);
}
#[cfg(windows)]
unsafe fn ignore_sigpipe() {}

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: int, argv: *const *const u8) -> int {
    // First and foremost we setup the stack. Usually this involves noting stack extents and
    // setting up overflow handlers.
    let _stack = unsafe {
        sys::stack::setup(true)
    };
    // Next, we store some thread specific information along with a nice name for the thread.
    thread_info::set(thread_info::NewThread::new(Some("<main>".to_string())));
    let exit_code = unsafe {
        // Store our args if necessary in a squirreled away location
        args::init(argc, argv);
        // By default, some platforms will send a *signal* when a EPIPE error
        // would otherwise be delivered. This runtime doesn't install a SIGPIPE
        // handler, causing it to kill the program, which isn't exactly what we
        // want!
        //
        // Hence, we set SIGPIPE to ignore when the program starts up in order
        // to prevent this problem.
        ignore_sigpipe();
        // And, finally, run the user code.
        match unwind::try(||{ mem::transmute::<*const u8, fn()>(main)(); }) {
            Ok(_) => env::get_exit_status() as isize,
            Err(_) => rt::DEFAULT_ERROR_CODE
        }
    };
    unsafe { cleanup(); }
    exit_code
}

/// Enqueues a procedure to run when the runtime is cleaned up
///
/// The procedure passed to this function will be executed as part of the
/// runtime cleanup phase. For normal rust programs, this means that it will run
/// after all other threads have exited.
///
/// The procedure is *not* executed with a local `Thread` available to it, so
/// primitives like logging, I/O, channels, spawning, etc, are *not* available.
/// This is meant for "bare bones" usage to clean up runtime details, this is
/// not meant as a general-purpose "let's clean everything up" function.
///
/// It is forbidden for procedures to register more `at_exit` handlers when they
/// are running, and doing so will lead to a process abort.
pub fn at_exit<F:FnOnce()+Send+'static>(f: F) {
    at_exit_imp::push(Thunk::new(f));
}

/// One-time runtime cleanup.
///
/// This function is unsafe because it performs no checks to ensure that the
/// runtime has completely ceased running. It is the responsibility of the
/// caller to ensure that the runtime is entirely shut down and nothing will be
/// poking around at the internal components.
///
/// Invoking cleanup while portions of the runtime are still in use may cause
/// undefined behavior.
pub unsafe fn cleanup() {
    args::cleanup();
    // FIXME: (#20012): the resources being cleaned up by at_exit
    // currently are not prepared for cleanup to happen asynchronously
    // with detached threads using the resources; for now, we leak.
    // at_exit_imp::cleanup();
}
