// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(start)]

use std::ffi;
use std::process::{Command, Output};
use std::rt::unwind::try;
use std::str;

#[start]
fn start(argc: isize, argv: *const *const u8) -> isize {
    if argc > 1 {
        unsafe {
            match **argv.offset(1) {
                1 => {}
                2 => println!("foo"),
                3 => assert!(try(|| {}).is_ok()),
                4 => assert!(try(|| panic!()).is_err()),
                5 => assert!(Command::new("test").spawn().is_err()),
                _ => panic!()
            }
        }
        return 0
    }

    let me = unsafe {
        str::from_utf8(ffi::CStr::from_ptr(*argv as *const i8).to_bytes()).unwrap()
    };
    let x = &"\x01";
    pass(Command::new(me).arg(x).output().unwrap());
    let x = &"\x02";
    pass(Command::new(me).arg(x).output().unwrap());
    let x = &"\x03";
    pass(Command::new(me).arg(x).output().unwrap());
    let x = &"\x04";
    pass(Command::new(me).arg(x).output().unwrap());
    let x = &"\x05";
    pass(Command::new(me).arg(x).output().unwrap());

    0
}

fn pass(output: Output) {
    if !output.status.success() {
        println!("{:?}", str::from_utf8(&output.stdout));
        println!("{:?}", str::from_utf8(&output.stderr));
    }
}
