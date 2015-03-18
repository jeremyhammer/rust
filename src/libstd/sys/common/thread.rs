// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use boxed::Box;
use libc;
use thunk::Thunk;
use sys::stack;

/// The starting point of Rust threads. This sets up the stack, extracts the function to run and
/// invokes that.
#[no_stack_check]
pub fn start_thread(main: *mut libc::c_void) {
    unsafe {
        let _stack = stack::setup(false);
        let f: Box<Thunk> = Box::from_raw(main as *mut Thunk);
        f.invoke(());
    }
}
