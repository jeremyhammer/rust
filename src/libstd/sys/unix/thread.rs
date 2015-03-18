// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use cmp;
use ffi::CString;
use io;
use libc;
use mem;
use ptr;
use sys::os;
use thunk::Thunk;
use time::Duration;

use sys::stack;
use sys_common::thread::*;

pub type rust_thread = libc::pthread_t;

pub unsafe fn create(stack: usize, p: Thunk) -> io::Result<rust_thread> {
    let p = box p;
    let mut native: libc::pthread_t = mem::zeroed();
    let mut attr: libc::pthread_attr_t = mem::zeroed();
    assert_eq!(pthread_attr_init(&mut attr), 0);

    // Reserve room for the red zone, the runtime's stack of last resort.
    let stack_size = cmp::max(stack, stack::RED_ZONE + stack::min_stack_size(&attr) as usize);
    match pthread_attr_setstacksize(&mut attr, stack_size as libc::size_t) {
        0 => {}
        n => {
            assert_eq!(n, libc::EINVAL);
            // EINVAL means |stack_size| is either too small or not a
            // multiple of the system page size.  Because it's definitely
            // >= PTHREAD_STACK_MIN, it must be an alignment issue.
            // Round up to the nearest page and try again.
            let page_size = os::page_size();
            let stack_size = (stack_size + page_size - 1) &
                             (-(page_size as isize - 1) as usize - 1);
            assert_eq!(pthread_attr_setstacksize(&mut attr,
                                                 stack_size as libc::size_t), 0);
        }
    };

    let ret = pthread_create(&mut native, &attr, thread_start,
                             &*p as *const _ as *mut _);
    assert_eq!(pthread_attr_destroy(&mut attr), 0);

    return if ret != 0 {
        Err(io::Error::from_os_error(ret))
    } else {
        mem::forget(p); // ownership passed to pthread_create
        Ok(native)
    };

    #[no_stack_check]
    extern fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
        start_thread(main);
        0 as *mut _
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub unsafe fn set_name(name: &str) {
    // pthread_setname_np() since glibc 2.12
    // availability autodetected via weak linkage
    type F = unsafe extern fn(libc::pthread_t, *const libc::c_char)
                              -> libc::c_int;
    extern {
        #[linkage = "extern_weak"]
        static pthread_setname_np: *const ();
    }
    if !pthread_setname_np.is_null() {
        let cname = CString::new(name).unwrap();
        mem::transmute::<*const (), F>(pthread_setname_np)(pthread_self(),
                                                           cname.as_ptr());
    }
}

#[cfg(any(target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd"))]
pub unsafe fn set_name(name: &str) {
    extern {
        fn pthread_set_name_np(tid: libc::pthread_t, name: *const libc::c_char);
    }
    let cname = CString::new(name).unwrap();
    pthread_set_name_np(pthread_self(), cname.as_ptr());
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub unsafe fn set_name(name: &str) {
    extern {
        fn pthread_setname_np(name: *const libc::c_char) -> libc::c_int;
    }
    let cname = CString::new(name).unwrap();
    pthread_setname_np(cname.as_ptr());
}

pub unsafe fn join(native: rust_thread) {
    assert_eq!(pthread_join(native, ptr::null_mut()), 0);
}

pub unsafe fn detach(native: rust_thread) {
    assert_eq!(pthread_detach(native), 0);
}

pub unsafe fn yield_now() {
    assert_eq!(sched_yield(), 0);
}

pub fn sleep(dur: Duration) {
    unsafe {
        if dur < Duration::zero() {
            return yield_now()
        }
        let seconds = dur.num_seconds();
        let ns = dur - Duration::seconds(seconds);
        let mut ts = libc::timespec {
            tv_sec: seconds as libc::time_t,
            tv_nsec: ns.num_nanoseconds().unwrap() as libc::c_long,
        };
        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        while dosleep(&mut ts) == -1 {
            assert_eq!(os::errno(), libc::EINTR);
        }
    }

    #[cfg(target_os = "linux")]
    unsafe fn dosleep(ts: *mut libc::timespec) -> libc::c_int {
        extern {
            fn clock_nanosleep(clock_id: libc::c_int, flags: libc::c_int,
                               request: *const libc::timespec,
                               remain: *mut libc::timespec) -> libc::c_int;
        }
        clock_nanosleep(libc::CLOCK_MONOTONIC, 0, ts, ts)
    }
    #[cfg(not(target_os = "linux"))]
    unsafe fn dosleep(ts: *mut libc::timespec) -> libc::c_int {
        libc::nanosleep(ts, ts)
    }
}


extern {
    fn pthread_self() -> libc::pthread_t;
    fn pthread_create(native: *mut libc::pthread_t,
                      attr: *const libc::pthread_attr_t,
                      f: extern fn(*mut libc::c_void) -> *mut libc::c_void,
                      value: *mut libc::c_void) -> libc::c_int;
    fn pthread_join(native: libc::pthread_t,
                    value: *mut *mut libc::c_void) -> libc::c_int;
    fn pthread_attr_init(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    fn pthread_attr_destroy(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                 stack_size: libc::size_t) -> libc::c_int;
    fn pthread_detach(thread: libc::pthread_t) -> libc::c_int;
    fn sched_yield() -> libc::c_int;
}
