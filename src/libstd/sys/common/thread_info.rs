// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // stack_guard isn't used right now on all platforms

use core::prelude::*;

use cell::RefCell;
use string::String;
use thread::Thread;

thread_local! { static THREAD_INFO: RefCell<Option<Thread>> = RefCell::new(None) }

pub fn current_thread() -> Option<Thread> {
    THREAD_INFO.with(|cell| cell.borrow().clone())
}

pub fn set(thread: Thread) {
    THREAD_INFO.with(|c| assert!(c.borrow().is_none()));
    THREAD_INFO.with(move |c| *c.borrow_mut() = Some(thread));
}

// a hack to get around privacy restrictions; implemented by `std::thread`
pub trait NewThread {
    fn new(name: Option<String>) -> Self;
}
